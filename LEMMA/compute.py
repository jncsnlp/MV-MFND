import ast
with open("/home/jncsnlp4/tb/LEMMA-main/out/prediction_twitter_llava.txt",'r',encoding='utf-8') as file:
    prediction = file.read()
    # file.write(str(fake_logits))
with open("/home/jncsnlp4/tb/LEMMA-main/out/twitter_labels.txt",'r',encoding='utf-8') as file:
    labels = file.read()
    # file.write(str(labels))
prediction = ast.literal_eval(prediction)
labels = ast.literal_eval(labels)    

print(len(prediction))
print(len(labels))
tp = 0
tn = 0
fp = 0
fn = 0
for x in range(len(prediction)):
    if prediction[x] == labels[x]:
        if labels[x] == 0:
            tp = tp + 1
        else:
            tn = tn + 1
    else:
        if labels[x] == 0:
            fp = fp + 1
        else:
            fn = fn + 1  

print(f"accuracy:{(tp+tn)/(tp+tn+fp+fn)}")
print(f"precision:{tp/(tp+fp)}")
print(f"recall:{tp/(tp+fn)}")
print(f"f1:{2*((tp/(tp+fp))*(tp/(tp+fn)))/(tp/(tp+fp)+tp/(tp+fn))}")            