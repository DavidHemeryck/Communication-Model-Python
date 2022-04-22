# TODO Install necessary packages via: conda install --file requirements.txt

import os
from io import StringIO

# if using anaconda3 and error execute: conda install --channel conda-forge pillow=5.2.0
import numpy as np

import huffman
import lzw
import util
from channel import channel
from imageSource import ImageSource
from unireedsolomon import rs
from util import Time

# ========================= SOURCE =========================
# TODO Select an image

IMG_NAME = 'ibe_op_de_grond.jpg'

dir_path = os.path.dirname(os.path.realpath(__file__))
IMG_PATH = os.path.join(dir_path, IMG_NAME)  # use absolute path

print(F"Loading {IMG_NAME} at {IMG_PATH}")
img = ImageSource().load_from_file(IMG_PATH)
input_lzw = img.get_pixel_seq().copy()

# print(img)
# uncomment if you want to display the loaded image
# img.show()
# uncomment if you want to show the histogram of the colors
# img.show_color_hist()


# ================================================================

# ======================= SOURCE ENCODING ========================
# =========================== Huffman ============================

# Use t.tic() and t.toc() to measure the executing time as shown below

t = Time()

# TODO Determine the number of occurrences of the source or use a fixed huffman_freq
"""
t.tic()
pixel_seq = ImageSource.get_pixel_seq(img)
pixel_value, count = np.unique(pixel_seq, return_counts=True)

huffman_freq = np.asarray((pixel_value, count)).T

huffman_tree = huffman.Tree(huffman_freq)
print(F"Generating the Huffman Tree took {t.toc_str()}")
t.tic()
# TODO print-out the codebook and validate the codebook (include your findings in the report)
encoded_message = huffman.encode(huffman_tree.codebook, img.get_pixel_seq())
print(huffman_tree.codebook)
print(len(encoded_message))
print("Enc: {}".format(t.toc()))

t.tic()
decoded_message = huffman.decode(huffman_tree, encoded_message)
print("Dec: {}".format(t.toc()))
"""


# ======================= SOURCE ENCODING ========================
# ====================== Lempel-Ziv-Welch ========================
def Lempel_Ziv_Welch_Encoding(input_lzw_bit):
    input_lzw_uint8 = util.bit_to_uint8(input_lzw_bit)
    t.tic()
    encoded_msg, dictonary = lzw.encode(input_lzw_uint8)
    print("Enc: {}".format(t.toc()))
    encoded_msg_bit = util.uint16_to_bit(encoded_msg)
    return encoded_msg_bit


def Lempel_Ziv_Welch_Decoding(encoded_message):
    encoded_message_int = util.bit_to_uint16(encoded_message).tolist()
    t.tic()
    decoded_msg = lzw.decode(encoded_message_int)
    print("Dec: {0:.4f}".format(t.toc()))
    decoded_msg_uint8 = np.array(decoded_msg, dtype=np.uint8)
    decoded_msg_bit = util.uint8_to_bit(decoded_msg_uint8)
    return decoded_msg_bit


# ====================== CHANNEL ENCODING ========================
# ======================== Reed-Solomon ==========================
n = 255  # code_word_length in symbols
k = 223  # message_length in symbols
coder = rs.RSCoder(n, k)


def Reed_Solomon_Encoding(encoded_msg_bit):
    # as we are working with symbols of 8 bits
    # choose n such that m is divisible by 8 when n=2^m−1
    # Example: 255 + 1 = 2^m -> m = 8

    # TODO generate a matrix with k symbols per rows (for each message)
    # TODO afterwards you can iterate over each row to encode the message

    encoded_message_uint8 = util.bit_to_uint8(encoded_msg_bit)

    # add extra zeros to make the array the correct length for reshaping into a matrix
    number_of_added_zeros = (int)(np.ceil(len(encoded_message_uint8) / k) * k - len(encoded_message_uint8))
    padded_encoded_msg = np.concatenate([encoded_message_uint8, np.zeros(number_of_added_zeros, dtype=np.uint8)])

    messages = np.reshape(padded_encoded_msg, (-1, k))
    rs_encoded_message = StringIO()
    print("input length: ", messages.shape)

    t.tic()
    for message in messages:
        code = coder.encode_fast(message, return_string=True)
        rs_encoded_message.write(code)
    t.toc_print()
    # TODO What is the RSCoder outputting? Convert to a uint8 (byte) stream before putting it over the channel
    rs_encoded_message_uint8 = np.array(
        [ord(c) for c in rs_encoded_message.getvalue()], dtype=np.uint8)

    rs_encoded_msg_bit = util.uint8_to_bit(rs_encoded_message_uint8)
    print(type(rs_encoded_msg_bit))

    return rs_encoded_msg_bit, number_of_added_zeros


def encode_rs(message):
    t = Time()
    byte_arr = util.bit_to_uint8(message)

    # aanvullen zodat matrix kan gevormd worden met bepaalde dimensie
    numpadd = (int)(np.ceil(len(byte_arr) / k) * k - len(byte_arr))
    byte_arr = np.append(byte_arr, np.zeros(numpadd))

    byte_matrix = np.array(byte_arr).reshape((-1, k))
    rs_encoded_message = StringIO()

    t.tic()
    # bericht per bericht encoderen
    for message in byte_matrix:
        code = coder.encode_fast(message, return_string=True)
        rs_encoded_message.write(code)
    tijd = t.toc()
    print("enc rs: ", tijd)
    rs_encoded_message_uint8 = np.array(
        [ord(c) for c in rs_encoded_message.getvalue()], dtype=np.uint8)

    return (util.uint8_to_bit(rs_encoded_message_uint8), numpadd, tijd)


def Channel(rs_encoded_message_bit):
    t.tic()
    received_message_bit = channel(rs_encoded_message_bit, ber=0.55)
    print("Channel: {}".format(t.toc()))
    return received_message_bit[0]


def Reed_Solomon_Decoding(received_message_bit, rs_encoded_msg_bit):
    # TODO Use this helper function to convert a bit stream to a uint8 stream
    received_message_uint8 = util.bit_to_uint8(received_message_bit)
    encoded_message_uint8 = util.bit_to_uint8(rs_encoded_msg_bit)

    decoded_message = StringIO()

    matrix1 = np.reshape(received_message_uint8,
                         (-1, n))
    print(matrix1.shape)

    number_of_added_zeros = k - int((len(encoded_message_uint8) % k))
    padded_encoded_msg = np.concatenate([encoded_message_uint8, np.zeros(number_of_added_zeros, dtype=np.uint8)])
    print("padded og: ", len(padded_encoded_msg))

    messages = np.reshape(padded_encoded_msg, (-1, k))

    # matrix2 = np.reshape(rs_encoded_message_uint8,
    #                    (-1, n))

    t.tic()
    # TODO Iterate over the received messages and compare with the original RS-encoded messages
    for cnt, (block, original_block) in enumerate(
            zip(matrix1, messages)):
        try:
            decoded, ecc = coder.decode_fast(block, return_string=True)
            assert coder.check(decoded + ecc), "Check not correct"
            decoded_message.write(str(decoded))
        except rs.RSCodecError as error:
            diff_symbols = len(block) - (original_block == block).sum()
            print(
                F"Error occured after {cnt} iterations of {len(encoded_message_uint8)}")
            print(F"{diff_symbols} different symbols in this block")
    t.toc_print()

    rs_decoded_message_uint8 = np.array(
        [ord(c) for c in decoded_message.getvalue()], dtype=np.uint8)
    rs_decoded_msg_bit = util.uint8_to_bit(rs_decoded_message_uint8)
    print("rec message: ", len(rs_decoded_message_uint8))
    print("DECODING COMPLETE")
    return rs_decoded_msg_bit


"""
def decode_rs(received_message, rs_encoded_message, numpadd):
    decoded_message = StringIO()

    t = Time()
    received_message_uint8 = util.bit_to_uint8(received_message)
    matrix1 = np.array(
        received_message_uint8).reshape((-1, n))

    rs_encoded_message_uint8 = util.bit_to_uint8(rs_encoded_message)
    rs_encoded_message_uint8 = np.append(
        rs_encoded_message_uint8, np.zeros(numpadd))

    rs_encoded_message_uint8_matrix = np.array(
        rs_encoded_message_uint8).reshape((-1, k))

    t.tic()
    # stap voor stap decoderen
    for cnt, (block, original_block) in enumerate(zip(matrix1, rs_encoded_message_uint8_matrix)):
        try:
            decoded, ecc = coder.decode_fast(block, return_string=True)
            assert coder.check(decoded + ecc), "Check not correct"
            decoded_message.write(str(decoded))
        except rs.RSCodecError as error:
            diff_symbols = len(block) - (original_block == block).sum()
            print(
                F"Error occured after {cnt} iterations of {len(received_message_uint8)}")
            print(F"{diff_symbols} different symbols in this block")
    tijd = t.toc()
    print("Dec rs: ", tijd)

    decoded_message_uint8 = np.array(
        [ord(c) for c in decoded_message.getvalue()], dtype=np.uint8)

    return util.uint8_to_bit(decoded_message_uint8[:-numpadd]), tijd
"""

# TODO after everything works, try to simulate the communication model as specified in the assignment

input_uint8 = img.get_pixel_seq()

input_bit = util.uint8_to_bit(input_uint8)

encoded_message_bit = Lempel_Ziv_Welch_Encoding(input_bit)

# rs_encoded_message_bit, number_zeros = Reed_Solomon_Encoding(input_bit)

# rs_encoded_message_bit, number_of_zeros = Reed_Solomon_Encoding(input_bit)
rs_encoded_message_bit, number_of_zeros,tijd = encode_rs(input_bit)

# received_message_bit = Channel(rs_encoded_message_bit)

rs_decoded_message_bit = Reed_Solomon_Decoding(rs_encoded_message_bit, input_bit)
# rs_decoded_message_bit = decode_rs(rs_encoded_message_bit, input_bit, number_of_zeros)

# rs_decoded_message_bit = rs_decoded_message_bit[:-number_of_zeros]

print(rs_decoded_message_bit == input_bit)
print(len(rs_decoded_message_bit))
print(len(input_bit))

decoded_message_bit = Lempel_Ziv_Welch_Decoding(encoded_message_bit)
