def common_prefix(strings):
    if not strings:
        return ""
    prefix = strings[0]
    for value in strings[1:2]:
        while not value.startswith(prefix):
            prefix = prefix[:-1]
    return prefix
