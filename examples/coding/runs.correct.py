def runs(text):
    result = []
    for character in text:
        if result and result[-1][0] == character:
            result[-1][1] += 1
        else:
            result.append([character, 1])
    return result
