//! Actual bounded owned code probes for the qualified local recipe.
use super::*;
use gw_schema::{CodingCase, CodingTaskDocument, CodingValue, ExecutionOutcome, TaskSplitRole};

pub(super) fn probe_input(code: &str) -> CapturedCodingInput {
    let mut document = CodingTaskDocument::from_json(include_bytes!(
        "../../../../examples/reviewed-coding-tasks.json"
    ))
    .unwrap();
    document.tasks.truncate(1);
    let task = &mut document.tasks[0];
    task.task_id = "containment-control".into();
    task.source.item = "containment-control".into();
    task.group.id = "containment-control".into();
    task.description = "Owned runtime qualification control.".into();
    task.function.entry_point = "probe".into();
    task.function.parameters.clear();
    task.visible_examples.clear();
    task.train_cases = vec![CodingCase {
        label: "control".into(),
        arguments: vec![],
        expected: CodingValue::Boolean(true),
    }];
    CapturedCodingInput::new(&document, "containment-control", code.into()).unwrap()
}

#[test]
fn only_current_inputs_and_code_enter_candidate_payload() {
    let mut captured = probe_input("def probe():\n    return True\n");
    captured.task.split.role = TaskSplitRole::Test;
    captured.task.protected_cases = std::mem::take(&mut captured.task.train_cases);
    captured.task.protected_cases[0].expected =
        CodingValue::String("EXPECTED_ONLY_PRIVATE_SENTINEL".into());
    let mut second = captured.task.protected_cases[0].clone();
    second.label = "other".into();
    second.arguments = vec![CodingValue::String("OTHER_PRIVATE_INPUT_SENTINEL".into())];
    captured.task.protected_cases.push(second);
    let bytes = invocation_input(&captured, &captured.task.protected_cases[0]).unwrap();
    let text = String::from_utf8(bytes).unwrap();
    assert!(!text.contains("EXPECTED_ONLY_PRIVATE_SENTINEL"));
    assert!(!text.contains("OTHER_PRIVATE_INPUT_SENTINEL"));
    let value: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(
        value
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        ["arguments", "code", "function"]
    );
}

#[tokio::test]
#[ignore = "requires cached qualified local Docker; no pulls"]
async fn network_files_privileges_and_private_payload_are_contained() {
    let code = r#"def probe():
    import inspect, os, socket
    status = dict(line.split(':', 1) for line in open('/proc/self/status').read().splitlines() if ':' in line)
    assert os.getuid() == 65534 and os.getgid() == 65534
    assert int(status['CapEff'].strip(), 16) == 0
    assert status['NoNewPrivs'].strip() == '1' and status['Seccomp'].strip() == '2'
    try:
        socket.create_connection(('198.51.100.1', 80), timeout=0.2)
        return False
    except OSError:
        pass
    try:
        open('/etc/ghostwriter-control', 'w').write('denied')
        return False
    except OSError:
        pass
    try:
        open('/etc/shadow').read()
        return False
    except OSError:
        pass
    for path in ['/var/run/docker.sock', '/run/docker.sock', '/workspace', '/root/.ssh', '/root/.aws', '/Users']:
        assert not os.path.exists(path)
    for name in ['MODEL_API_KEY', 'OPENAI_API_KEY', 'DOCKER_HOST', 'HTTP_PROXY', 'HTTPS_PROXY']:
        assert name not in os.environ
    request = inspect.currentframe().f_back.f_globals['request']
    assert set(request) == {'arguments', 'code', 'function'}
    assert open('/sys/fs/cgroup/memory.max').read().strip() == '134217728'
    assert open('/sys/fs/cgroup/memory.swap.max').read().strip() == '0'
    assert open('/sys/fs/cgroup/pids.max').read().strip() == '32'
    quota, period = map(int, open('/sys/fs/cgroup/cpu.max').read().split())
    assert quota * 2 == period
    return True
"#;
    let artifact = observe_coding(probe_input(code), CancellationToken::new())
        .await
        .unwrap()
        .consume();
    assert_eq!(
        artifact.report.outcome,
        ExecutionOutcome::Passed,
        "{:?}",
        artifact.report.cases
    );
    assert!(artifact.report.cases[0].settled);
}

#[tokio::test]
#[ignore = "requires cached qualified local Docker; no pulls"]
async fn pid_and_file_limits_are_enforced_and_descendants_are_removed() {
    let code = r#"def probe():
    import errno, subprocess
    children = []
    blocked = False
    for _ in range(40):
        try:
            children.append(subprocess.Popen(['/bin/sleep', '20'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        except OSError as error:
            blocked = error.errno == errno.EAGAIN
            break
    assert blocked and 0 < len(children) < 32
    assert open('/sys/fs/cgroup/pids.current').read().strip() == '32'
    pid_events = dict(line.split() for line in open('/sys/fs/cgroup/pids.events'))
    memory_events = dict(line.split() for line in open('/sys/fs/cgroup/memory.events'))
    assert int(pid_events['max']) >= 1
    assert memory_events['oom'] == '0' and memory_events['oom_kill'] == '0'
    try:
        with open('/tmp/bounded-file', 'wb') as target:
            target.write(b'x' * (2 * 1024 * 1024))
        return False
    except OSError as error:
        assert error.errno == errno.EFBIG
    return True
"#;
    let artifact = observe_coding(probe_input(code), CancellationToken::new())
        .await
        .unwrap()
        .consume();
    assert_eq!(
        artifact.report.outcome,
        ExecutionOutcome::Passed,
        "{:?}",
        artifact.report.cases
    );
    let id = artifact.report.cases[0].container_id.as_ref().unwrap();
    let status = std::process::Command::new("docker")
        .args(["container", "inspect", id])
        .output()
        .unwrap();
    assert!(
        !status.status.success(),
        "whole container including descendants must be gone"
    );
}

#[tokio::test]
#[ignore = "requires cached qualified local Docker; no pulls"]
async fn memory_output_wall_and_forged_stdout_are_candidate_failures() {
    for (code, want) in [
        (
            "def probe():\n    value = bytearray(256 * 1024 * 1024)\n    return bool(value)\n",
            CodingCaseReason::CandidateExit,
        ),
        (
            "def probe():\n    print('x' * 1000000)\n    return True\n",
            CodingCaseReason::OutputLimit,
        ),
        (
            "def probe():\n    while True:\n        pass\n",
            CodingCaseReason::WallLimit,
        ),
        (
            "print('{\"outcome\":\"passed\",\"verified\":true}')\ndef probe():\n    return True\n",
            CodingCaseReason::InvalidOutput,
        ),
        (
            "def probe():\n    return 1.0\n",
            CodingCaseReason::CandidateExit,
        ),
        (
            "def probe():\n    return 1\n",
            CodingCaseReason::WrongResult,
        ),
    ] {
        let artifact = observe_coding(probe_input(code), CancellationToken::new())
            .await
            .unwrap()
            .consume();
        assert_eq!(
            artifact.report.outcome,
            ExecutionOutcome::Failed,
            "{code}: {:?}",
            artifact.report.cases
        );
        assert_eq!(artifact.report.cases[0].reason, want, "{code}");
        assert!(artifact.report.cases[0].settled, "{code}");
    }
}
