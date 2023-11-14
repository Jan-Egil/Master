a = open("test.txt", "w")

for i in range(35):
    for j in range(3):
        stringvar = "\includegraphics[width=0.25\linewidth]{Figures/clts_ults_pics_35label/label_"
        stringvar += str(i+1)
        stringvar += "_rank_"
        stringvar += str(j+1)
        stringvar += ".png}\n"
        a.write(stringvar)
