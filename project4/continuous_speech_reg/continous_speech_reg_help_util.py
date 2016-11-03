import mfcc
import readwave


def make_fake_mfcc(construct_record):
    """制作fake录音\n
    输入:fake录音的字符串描述\n
    输出:(label，fake_mfcc_sequence)元组\n
    e.g 输入123,输出(['1','2','3'],123的mfcc_sequence)"""
    label = []
    return_mfcc_sequence = []
    for digit in construct_record:
        label.append(digit)
        return_mfcc_sequence += mfcc.mfcc(readwave.read_wave("./fake/"+digit+".wav"), "fm")
    return label, return_mfcc_sequence
