from fense.evaluator import Evaluator

fense = Evaluator(device='cuda', sbert_model=None)
test_cases = [
    "a group of people are talking to each other and a rooster crows",
    "a door is opened and closed and closed",
    "rain is pouring down onto a metal roof",
    "a heavy rain is falling on a metal roof",
    "a squeaky door is opened and closed and closed",
    "a door is being opened and closed and closed",
    "someone is opening and closing a door and closing the door",
    "a group of people are talking and walking",
    "a machine is running at a constant speed",
    "a person is playing a song with a stringed instrument",
    "the wind is blowing while a plane is flying overhead"
]

fl_err = fense.detect_error_sents(test_cases, batch_size=4)
print(fl_err)

for err in fl_err:
    print (1 - 0.9 * err * 1)