import bminf
cpm2 = bminf.models.CPM2()

with open ("input.txt", "r") as f:
    text = f.read();

    for result in cpm2.fill_blank(text, 
        top_p=1.0,
        top_n=5, 
        temperature=0.5,
        frequency_penalty=0,
        presence_penalty=0
    ):

        value = result["text"]
        text = text.replace("<span>", value, 1)

    fp = open("output.txt", "w");
    print(text, file = fp)