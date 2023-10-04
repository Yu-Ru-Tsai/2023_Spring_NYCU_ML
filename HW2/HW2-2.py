
def choose_Nk(N, k):
    ans = 1
    for i in range(1,k+1):
        ans *=  (N+1-i)/i
    return ans

if __name__=='__main__':

    fp = open('testfile.txt', 'r')
    a = int(input('a: '))
    b = int(input('b: '))
    line = fp.readline()
    case = 1
    while line:
        positive = line.count('1')
        negative = line.count('0')
        total = positive + negative
        positiveProb = positive/total
        negativeProb = negative/total
        likelihood = choose_Nk(total, positive)*(positiveProb**positive)*(negativeProb**negative)
        print(f"case {case}: {line}", end='')
        print(f"Likelihood: {likelihood}")
        print(f"Beta prior:     a = {a} b = {b}")
        a += positive
        b += negative
        print(f"Beta posterior: a = {a} b = {b}")
        print('\n')
        line = fp.readline()
        case += 1
