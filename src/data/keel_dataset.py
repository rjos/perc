import numpy as np
import re


class KeelAttribute:
    TYPE_REAL, TYPE_INTEGER, TYPE_NOMINAL = range(3)

    def __init__(self, attribute_name, attribute_type, attribute_range, builder):
        self.name = attribute_name
        self.type = attribute_type
        self.range = attribute_range
        self.builder = builder


class KeelDataSet:
    UNKNOWN = '?'

    def __init__(self, relation_name, attributes, data, inputs=None, outputs=None):
        self.name = relation_name
        self.attributes = attributes
        self.data = data
        self.inputs = inputs
        self.outputs = outputs
        self.shape = len(data[0]), len(data)

    def get_data(self, attributes):
        return [self.data[self.attributes.index(a)] for a in attributes]

    def get_data_target(self, inputs=None, outputs=None):
        inputs = inputs or self.inputs
        outputs = outputs or self.outputs

        assert inputs and outputs, 'You should specify inputs and outputs either here or in the initialization.'

        return np.transpose(self.get_data(inputs)), np.concatenate(self.get_data(outputs))

    def describe(self):
        lbls = np.array(self.get_data(self.outputs)[0])
        classes, _ = np.unique(lbls, return_counts=True)

        return {'name': self.name, 'instances': self.shape, 'attributes': len(self.inputs), 'classes': classes.shape[0]}

    def get_header(self):
        header = '@relation ' + self.name
        attributes = []
        for a in self.attributes:
            a_t = 'integer' if a.type == KeelAttribute.TYPE_INTEGER else 'real' if a.type == KeelAttribute.TYPE_REAL else ''
            if a_t != '':
                attributes.append('@attribute ' + a.name + ' ' + a_t + ' [' + str(a.range[0]) + ',' + str(a.range[1]) + ']')
            else:
                attributes.append('@attribute ' + a.name + ' {' + a.range + '}')
        header += '\n'.join(attributes)
        header += '\n'
        header += '@inputs {}\n'.format(','.join([a.name for a in self.inputs]))
        header += '@outputs {}\n'.format(','.join([a.name for a in self.outputs]))
        header += '@data\n'
        return header

    def save(self, path):
        with open(path, 'w') as in_file:
            in_file.write(self.get_header())
            data = list(map(list, zip(*self.data)))
            for i, d in enumerate(data):
                data[i] = list(map(lambda x,y: x.builder(y), self.attributes, d))
            data = '\n'.join(map(''.join, map(lambda x: map(str, x), data)))
            in_file.write(data)


def __parse_attributes_list(l, lkp):
    ret = []
    for i in l[:-1]:
        ret.append(lkp[i[:-1]])
    ret.append(lkp[l[-1]])
    return ret


def load_from_file(file):
    is_str = type(file) == str
    handle = open(file) if is_str else file
    try:
        top = handle.readline()

        l = top.split()
        if l[0] != '@relation' or len(l) != 2:
            raise SyntaxError('This is not a valid keel database.')

        relation_name = l[1]

        line = handle.readline().strip()

        attrs = []
        lkp = {}
        while line.startswith('@attribute'):
            l = line.split(maxsplit=3)
            if len(l) == 3:
                spl = re.split(r'(\w+)(\[.*\])', l[2])
                if len(spl) == 4:
                    l[2] = spl[1]
                    l.append(spl[2])

            if len(l) != 4:
                raise NotImplementedError('This is probably a nominal parameter. We don\'t have support for this yet.',
                                          l, file)
            if l[2][0] != '{':
                name = l[1]
                a_type = l[2]
                a_range = l[3]
            else:
                l = line.split(maxsplit=2)
                name = l[1]
                a_type = 'nominal'
                a_range = l[2]

            if a_type == 'real':
                a_type = KeelAttribute.TYPE_REAL
                builder = float
            elif a_type == 'integer':
                a_type = KeelAttribute.TYPE_INTEGER
                builder = int
            elif a_type == 'nominal':
                a_type = KeelAttribute.TYPE_NOMINAL
                builder = str
            else:
                raise SyntaxError('Unknown type.')

            if a_type != KeelAttribute.TYPE_NOMINAL:
                a_min, a_max = a_range[1:-1].split(',')
                # print(file, a_range[1:-1])
                a_range = builder(a_min), builder(a_max)
            else:
                a_range = a_range[1:-1].replace(' ', '').split(',')

            k = KeelAttribute(name, a_type, a_range, builder)
            attrs.append(k)
            lkp[name] = k
            line = handle.readline().strip()

        l = line.split()
        if l[0] != '@inputs':
            raise SyntaxError('Expected @inputs.' + line + str(l))
        inputs = __parse_attributes_list(l[1:], lkp)

        line = handle.readline()
        l = line.split()

        if l[0] != '@outputs':
            raise SyntaxError('Expected @outputs.')

        outputs = __parse_attributes_list(l[1:], lkp)

        line = handle.readline()

        if line != '@data\n':
            raise SyntaxError('Expected @data.')

        data = [[] for _ in range(len(attrs))]
        for data_line in handle:
            if data_line:
                l = data_line.strip().split(',')
                for lst, value, attr in zip(data, l, attrs):
                    v = value
                    v = v if v == KeelDataSet.UNKNOWN else attr.builder(v)
                    lst.append(v)

        return KeelDataSet(relation_name, attrs, data, inputs, outputs)
    finally:
        if is_str:
            handle.close()
