from transformers import RobertaTokenizer, RobertaForQuestionAnswering
import torch

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForQuestionAnswering.from_pretrained("roberta-base")


print(model)

# question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
# inputs = tokenizer(question, text, return_tensors="pt")
# start_positions = torch.tensor([1])
# end_positions = torch.tensor([3])

# outputs = model(**inputs, start_positions=start_positions, end_positions=end_positions)
# loss = outputs.loss
# start_scores = outputs.start_logits
# end_scores = outputs.end_logits

# print(start_scores)
# print(end_scores)
