import numpy as np

def to_binary(seq):
    # eoncode non-standard amino acids like X as all zeros
    # output a array with size of L*20
    seq = seq.upper()
    aas = 'ACDEFGHIKLMNPQRSTVWY'
    pos = dict()
    for i in range(len(aas)): pos[aas[i]] = i
    
    binary_code = dict()
    for aa in aas: 
        code = np.zeros(21, dtype = np.int8)
        code[pos[aa]] = 1
        binary_code[aa] = code
    
    seq_coding = np.zeros((len(seq),21), dtype = np.int8)
    for i,aa in enumerate(seq): 
        code = binary_code.get(aa,np.zeros(21, dtype = np.int8))
        seq_coding[i,:] = code
    return seq_coding

def to_int(seq, max_length):
    seq.upper()
    d ={'A':1,
        'C':2,
        'D':3,
        'E':4,
        'F':5,
        'G':6,
        'H':7,
        'I':8,
        'K':9,
        'L':10,
        'M':11,
        'N':12,
        'P':13,
        'Q':14,
        'R':15,
        'S':16,
        'T':17,
        'V':18,
        'W':19,
        'Y':20,
        'X':21}
    tmp =np.array([d[i] for i in seq])
    out = np.zeros((max_length,))
    index = tmp.size if tmp.size<max_length else max_length
    out[:index] = tmp[:index]
    return out

def loss_weight(mask, dist, max_length):
    len_seq = len(mask)
    seq_w = [dist[i] for i in mask] 
    tmp = np.ones((max_length,))
    tmp[:len_seq] = seq_w
    tmp[len_seq:] = 0.0
    return tmp

def to_binary_mask(mask, typ=3):
    # eoncode non-standard amino acids like X as all zeros
    # output a array with size of L*20
    seq = mask.upper()
    aas_3 = 'BHC'
    aas_8 = 'HBEGITS-'
    if typ==3:
        aas = aas_3
    else:
        aas = aas_8
    l = len(aas)
    pos = dict()
    for i in range(len(aas)): pos[aas[i]] = i
    
    binary_code = dict()
    for aa in aas: 
        code = np.zeros(l, dtype = np.int8)
        code[pos[aa]] = 1
        binary_code[aa] = code
    
    seq_coding = np.zeros((len(seq),l), dtype = np.int8)
    for i, aa in enumerate(seq): 
        code = binary_code.get(aa,np.zeros(l, dtype = np.int8))
        seq_coding[i,:] = code
    return seq_coding

def mask_padding(inp, length=500):
    if len(inp) > length:
        return inp[:length]
    else:
        tmp = ''.join(['C' for _ in range(length - len(inp))])
        return(inp+tmp)
def zero_padding(inp,length=500,start=False):
    # zero pad input one hot matrix to desired length
    # start .. boolean if pad start of sequence (True) or end (False)
    #assert len(inp) <= length
    out = np.zeros((length,inp.shape[1]))
    if start:
        out[-inp.shape[0]:] = inp[:length,:]
    else:
        out[0:inp.shape[0]] = inp[:length,:]
    return out