import random


def random_create():
    """产生24个短电话号 26个长电话号，以 lables[] 返回"""
    random.seed(100)
    labels = []
    print("24个短的")
    for i in range(0, 24):
        number = []
        string = ""
        for j in range(0, 4):
            number.append(random.randint(0, 9))
            string += str(number[-1])
        print(i, ":  ", number)
        labels.append(string)
    print("26个长的")
    for i in range(0, 26):
        number = []
        string = ""
        for header in range(0, 3):
            number.append(random.randint(2, 9))
            string += str(number[-1])
        for tail in range(0, 4):
            number.append(random.randint(0, 9))
            string += str(number[-1])
        print(i, ":  ", number)
        labels.append(string)
    # write file
    txtfile = open('numbers.txt', 'w')
    for label in labels:
        txtfile.writelines(label + '\n')
    txtfile.close()
    return labels


if __name__ == '__main__':
    random_create()