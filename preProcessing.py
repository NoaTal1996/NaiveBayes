import pandas as pd
import statistics as stats
import glob


class preProcessing():
    def __init__(self, path, bins):
        self.path = path
        self.df_csv = pd.read_csv(path + '/train.csv', header=0)
        self.df_csv_test = pd.read_csv(path + '/test.csv', header=0)
        self.bins = bins
        self.classess = {}
        self.attr = []


    def prepros(self):
        lines = self.split_struture()
        labels = [i+1 for i in range(self.bins)]
        for line in lines:
            if line[2] == 'NUMERIC':
                self.df_csv[line[1]].fillna(self.df_csv[line[1]].mean(), inplace=True)
                self.df_csv_test[line[1]].fillna(self.df_csv[line[1]].mean(), inplace=True)
                self.attr.append(line[2])
            else:
                self.df_csv[line[1]].fillna(stats.mode(self.df_csv[line[1]]), inplace=True)
                self.df_csv_test[line[1]].fillna(stats.mode(self.df_csv[line[1]]), inplace=True)
                self.attr.append("Categorial")
        for line in lines:
            if line[2] == 'NUMERIC':
                self.df_csv[line[1]] = self.discretization(self.df_csv[line[1]], labels)
                self.df_csv_test[line[1]] = self.discretization(self.df_csv[line[1]], labels)
        self.count_vals_occurances()
        self.df_csv.to_csv("train_completet.csv")
        return self.df_csv

    def split_struture(self):
        structure = self.path + '/structure.txt'
        file = open(structure, 'r')
        content = file.readlines()
        lines = []
        for i in content:
            if '{' in i and '}' in i:
                splited = i.split('{')
                first_half = splited[0].split()
                second_half = splited[1].split('}')
                first_half.append(second_half[0].split(','))
                lines.append(first_half)
            else:
                lines.append(i.split())
        file.close()
        return lines

    def discretization(self, attribute, labels):
        min_val = attribute.min()
        max_val = attribute.max()
        w = (max_val-min_val)/self.bins
        bins = [((i+1)*w + min_val) for i in range(self.bins-1)]
        breakpoints = [min_val] + bins + [max_val]
        colbin = pd.cut(attribute, bins=breakpoints, labels=labels, include_lowest=True)
        return colbin


    def count_vals_occurances(self):
        vals = list(self.df_csv['class'].values)
        unique_vals = set(vals)
        for i in unique_vals:
            self.classess[i] = vals.count(i)



