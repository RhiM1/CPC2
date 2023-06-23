from evaluate import load

wer = load("wer")
predictions = ["this is the prediction", "there is an other sample"]

references = ["this is the reference q", "there is another one"]

wer_score = wer.compute(predictions=[predictions[1]], references=[references[1]])

print(wer_score)