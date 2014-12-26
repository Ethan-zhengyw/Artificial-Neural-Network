from nn import NetTable

def read_records(path):

    records = [record.split(',') for record in open(path).read().split('\n')
            if len(record.split(',')) == 65]

    for i in range(len(records)):
        for j in range(65):
            records[i][j] = int(records[i][j])

    return records

def test():

    hnn, alpha = 30, 0.05
    nt = NetTable(hnn, alpha)

    records_tra = read_records('../docs/digitstra.txt')
    records_tes = read_records('../docs/digitstest.txt')
    #records_tra = read_records('data.txt')
    #records_tes = read_records('data.txt')

    print "tranning..."
    for record in records_tra:
        for i in range(2):
            nt.back_propagate(record[:64], record[64])


    print "testing..."
    count = 0
    for record in records_tes:

        print "expect:", record[64],
        result = nt.feed_forward(record[:64])[1]
        index, value = 0, result[0]
        for ind, val in enumerate(result):
            if val > value:
                index = ind
        print "estimate:", index, result

        if record[64] == index:
            count = count + 1

    print "%d of %d" % (count, len(records_tes))

test()
