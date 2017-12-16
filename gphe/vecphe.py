import phe
from numba import jit
import numpy as np

@jit
def _chunks(l, n):
    """Split an array into blocks.

    Parameters
    ----------
    l : :obj:`list`
        The list of numbers
    n : int
        The block size

    Returns
    -------
    :obj:`list` of :obj:`list`s
        A list containing the blocks

    """
    for i in range(0, len(l), n):
        yield l[i:i + n]

def _frombase(v, base):
    """Return a number as an integer from its base representation

    Parameters
    ----------
    v : :obj:`list`
        The base representation of the number in a :obj:`list`
    base : int
        The base

    Returns
    -------
    N : int
        The number in base 10 represenation (as an integer)

    """
    N = 0
    B = 1
    for i in range(len(v)):
        N = N + v[i] * B
        B *= base
    return N

def _tobase(x, base, block):
    """Return the base representation of an integer

    Parameters
    ----------
    x : int
        The number to convert
    base : int
        The base

    Returns
    -------
    v : :obj:`list`
        The base representation of the number in a :obj:`list`

    """

    V = [0] * block
    for i in range(block):
        V[i] = x % base
        x = x // base
    return V

def vecEnc(pub, V):
    """Encrypt a vector of numbers using standard Pallier Encryption

    Parameters
    ----------
    pub : :obj:`PaillierPublicKey`
        The public key
    V : :obj:`list` of int
        The vector to encrypt

    Returns
    -------
    EV : :obj:`list` of :obj:`EncryptedNumber`
        The encrypted numbers

    """
    return [pub.encrypt(x) for x in V]

def vecDec(prv, EV):
    """Decrypt a vector of numbers using standard Pallier Encryption

    Parameters
    ----------
    prv : :obj:`PaillierPrivateKey`
        The private key
    EV : :obj:`list` of :obj:`EncryptedNumber`
        The vector to decrypt

    Returns
    -------
    V : :obj:`list` of int
        The decrypted numbers
    
    """
    return [prv.decrypt(x) for x in EV]

def fvecEnc(pub, V, base=128, block = 128):
    """Encrypt a vector of numbers using fast Pallier Encryption

    Parameters
    ----------
    pub : :obj:`PaillierPublicKey`
        The public key
    V : :obj:`list` of int
        The vector to encrypt
    base : int
        The base of the new number representations
    block : int
        The block size used to split V

    Returns
    -------
    Ns : :obj:`list` of :obj:`EncryptedNumber`
        The encrypted numbers, where each number represents a block

    """
    Vs = _chunks(V, block)
    Ns = []
    for V in Vs:
        N = _frombase(V, base)
        Ns.append(pub.encrypt(N))
    return Ns

def fvecDec(prv, E, base=128, block = 128):
    """Decrypt a vector of numbers using fast Pallier Encryption

    Parameters
    ----------
    prv : :obj:`PaillierPrivateKey`
        The private key
    EV : :obj:`list` of :obj:`EncryptedNumber`
        The vector to decrypt

    Returns
    -------
    V : :obj:`list` of int
        The decrypted numbers
    
    """
    EVs = [prv.decrypt(x) for x in E]
    V = []
    for EV in EVs:
        v = _tobase(EV, base, block)
        V.extend(v)
    return V

def matEnc(pub, M):
    """Encrypt a matrix of numbers using standard Pallier Encryption

    Parameters
    ----------
    pub : :obj:`PaillierPublicKey`
        The public key
    M : :obj:`ndarray` of shape (d x n) of int
        The matrix to encrypt

    Returns
    -------
    EM : :obj:`list` of :obj:`list` of :obj:`EncryptedNumber`
        The encrypted numbers matrix

    """
    M = M.T.tolist()
    return [vecEnc(pub, V) for V in M]

def fmatEnc(pub, M, base=128, block=128):
    """Encrypt a matrix of numbers using fast Pallier Encryption

    Parameters
    ----------
    pub : :obj:`PaillierPublicKey`
        The public key
    M : :obj:`ndarray` of shape (d x n) of int
        The matrix to encrypt
    base : int
        The base of the new number representations
    block : int
        The block size used to split V

    Returns
    -------
    EM : :obj:`list` of :obj:`list` of :obj:`EncryptedNumber`
        The encrypted numbers matrix

    """
    M = M.T.tolist()
    return [fvecEnc(pub, V, base=base, block=block) for V in M]

def matDec(prv, EM):
    """Decrypt a matrix of numbers using standard Pallier Encryption

    Parameters
    ----------
    prv : :obj:`PaillierPrivateKey`
        The private key
    EM : :obj:`list` of :obj:`list` of :obj:`EncryptedNumber` with shape 
         (d x n) where d is dimensions and n is samples

    Returns
    -------
    M : :obj:`ndarray` of int
        The decrypted matrix
    
    """
    M = [vecDec(prv, V) for V in EM]
    return np.array(M).astype(int).T

def fmatDec(prv, EM, base=128, block=128):
    """Decrypt a matrix of numbers using fast Pallier Encryption

    Parameters
    ----------
    prv : :obj:`PaillierPrivateKey`
        The private key
    EM : :obj:`list` of :obj:`list` of :obj:`EncryptedNumber` with shape 
         (d x n) where d is dimensions and n is samples

    Returns
    -------
    M : :obj:`ndarray` of int
        The decrypted matrix

    """
    M = [fvecDec(prv, V, base=base, block=block) for V in EM]
    return np.array(M).astype(int).T