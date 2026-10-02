"""Complete source, actual tokenizer, native reader, and real trainer dataloader evidence."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import pytest
from ghostwriter_trl.artifact import ContractError
from ghostwriter_trl.gemma31b.build import build, source_identity
from ghostwriter_trl.gemma31b.prepared import prepare, verify_prepared, save_prepared, read_prepared
from ghostwriter_trl.gemma31b.handoff import qualify_prepared_handoff
from ghostwriter_trl.gemma31b.source import project_messages
from ghostwriter_trl.gemma31b.projection import prepare_target
from ghostwriter_trl.gemma31b.policy import controls
from ghostwriter_trl.gemma31b import PROFILE
from ghostwriter_trl.prepared import _frame, _unframe, _json_bytes, _payload_json


def options(**changes):
    return dict(cot='masked', turns='all_assistant', max_length=4096,
                enable_thinking=True, preserve_thinking=True) | changes


def source_values(source):
    row = source.rows()[0]
    return json.loads(row['messages_json']), json.loads(row['tools_json'])


@pytest.mark.parametrize('cot', ['masked', 'supervised', 'stripped'])
@pytest.mark.parametrize('thinking', [False, True])
@pytest.mark.parametrize('preserve', [False, True])
def test_official_prefixes_native_roundtrip_and_shifted_labels(serial_source, tokenizer, gw, cot, thinking, preserve):
    data = prepare(serial_source, tokenizer, **options(cot=cot, enable_thinking=thinking, preserve_thinking=preserve))
    loaded = verify_prepared(data, gw, tokenizer)
    examples = loaded.examples
    assert [e['target_index'] for e in examples] == [2, 4, 6, 8]
    assert 'κλειδί value' in examples[0]['rendered']
    assert [e['target_kind'] for e in examples] == ['tool_call', 'tool_call', 'text_answer', 'text_answer']
    for e in examples:
        call = e['target_kind'] == 'tool_call'
        assert bool(e['shifted_call_token_indices']) == call
        assert bool(e['shifted_answer_token_indices']) != call
        assert e['input_ids'][-1:] == [50] if call else e['input_ids'][-2:] == [106, 107]
        assert e['labels'][-1:] == [50] if call else e['labels'][-2:] == [106, -100]
        assert 'RETAINED_RAW_ONLY' not in e['rendered']
        for span in e['spans']:
            if span['kind'] in {'definition', 'observation'} or span['message_index'] < e['target_index']:
                assert not span['supervised']
        for label, kinds in zip(e['labels'], e['token_kinds'], strict=True):
            if 'observation' in kinds or 'definition' in kinds:
                assert label == -100
        shifted = e['labels'][1:]
        for index in e['shifted_call_token_indices'] + e['shifted_answer_token_indices']:
            assert shifted[index-1] == e['input_ids'][index]
    assert ('Read the first observation.' in examples[-1]['rendered']) == (preserve and cot != 'stripped')
    assert loaded.manifest['effective_shifted_call_token_count'] == sum(len(e['shifted_call_token_indices']) for e in examples)


def test_repeated_capture_pyarrow_datasets_and_real_dataloader(serial_source, tokenizer, gw, tmp_path):
    from datasets import Dataset
    data = prepare(serial_source, tokenizer, **options())
    assert data == prepare(serial_source, tokenizer, **options())
    assert serial_source.rows()[0]['tools_json'] is not None
    assert len(Dataset.from_parquet(str(Path(__file__).parents[1] / 'fixtures/v5-gemma31b-serial.parquet'), cache_dir=str(tmp_path / 'datasets'))) == 1
    destination = tmp_path / 'prepared.gwsft'
    save_prepared(destination, data)
    with pytest.raises(FileExistsError):
        save_prepared(destination, data)
    loaded = read_prepared(destination, gw, tokenizer)
    destination.write_bytes(b'changed after capture')
    assert loaded.data == data
    detached = loaded.examples
    detached[0]['labels'][0] = 0
    assert loaded.examples[0]['labels'][0] == -100
    result = qualify_prepared_handoff(loaded, tokenizer)
    assert result['real_collator']['rows'] == result['real_sft_trainer_dataloader']['rows'] == 4
    assert result['real_sft_trainer_dataloader']['padding_masked']
    assert result['forward_passes'] == result['optimizer_steps'] == 0
    assert result['shifted_call_tokens'] > 0


def test_overlength_accounts_for_every_outcome_and_final_only(serial_source, tokenizer, gw):
    examples, _ = build(serial_source, tokenizer, **options())
    limit = len(examples[0]['input_ids'])
    loaded = verify_prepared(prepare(serial_source, tokenizer, **options(max_length=limit)), gw, tokenizer)
    assert [e['target_index'] for e in loaded.examples] == [2]
    assert [r['target_index'] for r in loaded.manifest['rejections']] == [4, 6, 8]
    assert loaded.manifest['candidate_target_count'] == 4
    assert loaded.examples[0]['rendered'].endswith('<|tool_response>')
    all_rejected = verify_prepared(prepare(serial_source, tokenizer, **options(max_length=1)), gw, tokenizer)
    assert all_rejected.examples == []
    assert len(all_rejected.manifest['rejections']) == 4
    final = verify_prepared(prepare(serial_source, tokenizer, **options(turns='final_turn_only')), gw, tokenizer)
    assert [e['target_index'] for e in final.examples] == [8]


@pytest.mark.parametrize('mutation', ['missing_definition','missing_reply','missing_id','duplicate_id','wrong_reply','parallel','content_and_call','multimodal','delimiter_argument','delimiter_key','delimiter_definition','ambiguous_key','schema_dropped','unanswered_suffix'])
def test_complete_source_rejection_before_prefix_selection(serial_source, mutation):
    m,t = source_values(serial_source)
    if mutation=='missing_definition': t=[]
    elif mutation=='missing_reply': del m[3]
    elif mutation=='missing_id': del m[2]['tool_calls'][0]['id']
    elif mutation=='duplicate_id': m[4]['tool_calls'][0]['id']=m[2]['tool_calls'][0]['id']
    elif mutation=='wrong_reply': m[3]['tool_call_id']='unknown'
    elif mutation=='parallel': m[2]['tool_calls'].append(deepcopy(m[2]['tool_calls'][0]))
    elif mutation=='content_and_call': m[2]['content']='Not empty'
    elif mutation=='multimodal': m[3]['content']=[{'type':'text','text':'x'}]
    elif mutation=='delimiter_argument': m[2]['tool_calls'][0]['function']['arguments']['query']='<|tool_response>'
    elif mutation=='delimiter_key': m[2]['tool_calls'][0]['function']['arguments']['<turn|>']=1
    elif mutation=='delimiter_definition': t[0]['function']['description']='<|tool_call>'
    elif mutation=='ambiguous_key': m[2]['tool_calls'][0]['function']['arguments']['query:injected']='x'
    elif mutation=='schema_dropped': t[0]['function']['parameters']['additionalProperties']=False
    elif mutation=='unanswered_suffix': m=m[:5]
    with pytest.raises(ContractError): project_messages(m,t,'masked')


@pytest.mark.parametrize('mutation', ['call_name','argument','observation_owner','handoff_owner','call_count','answer_count'])
def test_independently_rehashed_native_source_tampering_is_rejected(serial_source, tokenizer, gw, mutation):
    data=prepare(serial_source,tokenizer,**options())
    _,raw,source=_unframe(data)
    payload=_payload_json(raw)
    if mutation in {'call_name','argument'}:
        e=payload['examples'][0]
        old,new=('call:lookup','call:lookuq') if mutation=='call_name' else ('café','cafè')
        # For arguments change the actual call occurrence, not the preceding user text.
        position=e['rendered'].index('<|tool_call>')
        e['rendered']=e['rendered'][:position]+e['rendered'][position:].replace(old,new,1)
    elif mutation=='observation_owner':
        e=payload['examples'][1]
        next(s for s in e['spans'] if s['kind']=='observation')['kind']='context'
    elif mutation=='handoff_owner':
        payload['examples'][0]['spans'][-1]['kind']='call_wrapper'
    elif mutation=='call_count': payload['manifest']['effective_shifted_call_token_count']+=1
    elif mutation=='answer_count': payload['examples'][0]['shifted_answer_token_indices']=[1]
    # Recompute token kind declarations after changing span kind so stale bookkeeping
    # cannot be the reason a rehashed ownership edit is rejected.
    for e in payload['examples']:
        e['token_kinds'] = [sorted({s['kind'] for s in e['spans'] if s['start'] < end and s['end'] > start})
                            for start, end in e['ownership_offsets']]
    changed=_frame(_json_bytes(payload),source)
    result=subprocess.run([str(gw),'artifact','verify-prepared','--stdin'],input=changed,capture_output=True)
    assert result.returncode != 0
    with pytest.raises(ContractError): verify_prepared(changed,gw,tokenizer)


@pytest.mark.parametrize('case', json.loads((Path(__file__).parent / 'argument_cases.json').read_text()), ids=lambda c:c['name'])
def test_literal_numeric_domain_matches_official_template(serial_source,tokenizer,case):
    from ghostwriter_trl.gemma31b.source import source_json
    m,t=source_values(serial_source)
    def prepare_case():
        m[2]['tool_calls'][0]['function']['arguments']=source_json(case['raw'])
        projected=project_messages(m,t,'masked')
        return prepare_target(projected[:3],tokenizer,'masked',4096,tools=t,
                              settings=controls(PROFILE,True,True))
    if case['rendered'] is None:
        with pytest.raises(ContractError): prepare_case()
    else:
        example=prepare_case()
        assert '<|tool_call>call:lookup'+case['rendered']+'<tool_call|><|tool_response>' in example['rendered']


def test_captured_record_rejection_preserves_invalid_suffix_and_numeric_scope(tokenizer,gw):
    from ghostwriter_trl.artifact import read_snapshot
    snapshot=read_snapshot(Path(__file__).parents[1]/'fixtures/v5-gemma31b-rejected.parquet',gw)
    loaded=verify_prepared(prepare(snapshot,tokenizer,**options()),gw,tokenizer)
    assert loaded.examples==[]
    assert loaded.manifest['candidate_target_count']==0
    assert loaded.manifest['rejected_record_count']==3
    assert {r['record_id'] for r in loaded.manifest['rejections']}=={'incomplete-suffix','scientific-domain','parallel-calls'}
    assert all(r['target_index'] is None for r in loaded.manifest['rejections'])


def test_nested_definition_ledger_matches_literal_and_official_template(serial_source,tokenizer):
    m,t=source_values(serial_source)
    case=json.loads((Path(__file__).parent/'definition_cases.json').read_text())[0]
    t.append(case['tool'])
    projected=project_messages(m,t,'masked')
    example=prepare_target(projected[:3],tokenizer,'masked',4096,tools=t,settings=controls(PROFILE,True,True))
    assert case['rendered'] in example['rendered']
    assert all(not s['supervised'] for s in example['spans'] if s['kind']=='definition')


def test_mixed_unsupported_schema_types_keep_valid_record_targets(tokenizer,gw):
    from ghostwriter_trl.artifact import read_snapshot
    snapshot=read_snapshot(Path(__file__).parents[1]/'fixtures/v5-gemma31b-mixed-schema.parquet',gw)
    loaded=verify_prepared(prepare(snapshot,tokenizer,**options()),gw,tokenizer)
    assert len(loaded.examples)==4
    assert {e['source']['record_id'] for e in loaded.examples}=={'valid-source'}
    assert loaded.manifest['rejected_record_count']==3
    assert loaded.manifest['candidate_target_count']==4
    assert {r['record_id'] for r in loaded.manifest['rejections']}=={'array-type','object-type','union-type'}
    assert all(r['target_index'] is None for r in loaded.manifest['rejections'])


def test_verified_generated_bare_negative_zero_survives_native_and_official_replay(tokenizer,gw):
    from ghostwriter_trl.artifact import read_snapshot
    snapshot=read_snapshot(Path(__file__).parents[1]/'fixtures/v5-gemma31b-negative-zero.parquet',gw)
    assert '"negative":-0,' in snapshot.rows()[0]['messages_json']
    loaded=verify_prepared(prepare(snapshot,tokenizer,**options()),gw,tokenizer)
    assert len(loaded.examples)==4
    assert 'negative:-0.0' in loaded.examples[0]['rendered']
    assert loaded.manifest['rejected_item_count']==0


@pytest.mark.parametrize('case',json.loads((Path(__file__).parent/'identifier_cases.json').read_text()),ids=lambda c:repr(c['value']))
def test_identifier_ascii_domain_is_shared(case):
    from ghostwriter_trl.gemma31b.source import identifier
    if case['accepted']:
        identifier(case['value'])
    else:
        with pytest.raises(ContractError): identifier(case['value'])
