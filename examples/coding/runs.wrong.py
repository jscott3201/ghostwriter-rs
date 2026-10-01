def runs(text):
    from collections import Counter
    return [[character, count] for character, count in Counter(text).items()]
