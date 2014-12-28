#-*- coding: utf-8 -*-
from pprint import pprint
from nn import NetTable

def read_records(path):

    records = [record.split(',') for record in open(path).read().split('\n')
            if len(record.split(',')) == 65]

    for i in range(len(records)):
        for j in range(65):
            records[i][j] = int(records[i][j])

    return records

records_tra = read_records('../docs/digitstra.txt')
records_tes = read_records('../docs/digitstest.txt')

def test(nh, alpha, beta):

    #nh, alpha, beta = 200, 0.05, 0.2 的时候正确率可以达到92%
    nt = NetTable(nh, alpha, beta)
    nt.train(records_tra)

    count = 0
    for record in records_tes:

        result = nt.feed_forward(record[:64])[1]

        maxindex, maxvalue = 0, max(result)
        for index in range(10):
            if result[index] == max(result):
                maxindex = index
        
        if maxindex == record[64]:
            count = count + 1

    print "%d,%.4f,%.4f,%.4f" % (nh, alpha, beta, (float(count) / len(records_tes)))


def trainparam():
    for nh in range(50, 300):
        for alpha in range(5, 50, 1):
            for beta in range(1, 50, 1):
                test(nh, float(alpha)/100, float(beta)/100)
    #for i in range(10):
    #    test(200, 0.05, 0.1)


if __name__ == "__main__":
    trainparam()
