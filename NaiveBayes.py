import pandas as pd


class naiveBayes():

    def __init__(self):
        self.m_estimators = 2
    #the pipeline of classify of naive Bayes
    def classify(self, train_set, test_set, classes, attr, output):
        out = []
        i = 1
        for line in test_set.values:
            prob_dict = {}
            for c in classes.keys():
                prob = self.calc_prob(c, line, train_set, classes)
                ci = classes[c] / sum(list(classes.values()))
                prob_dict[c] = prob * ci
            max_prob = max(prob_dict, key=prob_dict.get)
            out.append(str(i) + ' ' + str(max_prob) + '\n')
            i += 1
        file = open(output, 'w')
        file.writelines(out)
        file.close()

    #calculator of m-estimator
    def calc_prob(self, clas, line, train, classes):
        prob_mult = 1
        cols = list(train.columns)
        for i in range(len(line) - 1):
            temp = train[train[cols[i]] == line[i]]
            M = len(train[cols[i]].unique())
            nc = len(temp[temp['class'] == clas])
            numerator = nc + (self.m_estimators * 1 / M)
            denumerator = self.m_estimators + classes[clas]
            prob = numerator / denumerator
            prob_mult *= prob
        return prob_mult
