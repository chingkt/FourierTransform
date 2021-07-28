import time
import numpy as np
from matplotlib import pyplot as plt
from scipy import fft
from scipy.linalg import dft


def dft_recursive(signal: np.ndarray):
    """
    Performs recursive dft using Divide and Conquer principle.

    :param signal: np.ndarray
    :return: transformed signal: np.ndarray

    """

    n = len(signal)
    if n == 1:
        return signal
    else:
        m = n // 2

        # recursively call the function
        # create two arrays to store values of even and odd index respectively
        even = dft_recursive(signal[0::2])
        odd = dft_recursive(signal[1::2])

        omega = np.exp(-2j * np.pi / n)
        z = np.empty((n,), dtype=np.complex128)

        # calculate current z
        for i in range(m):
            z[i] = even[i] + (omega ** i) * odd[i]
            z[i + m] = even[i] - (omega ** i) * odd[i]
        return z


def plotTime():
    """
    Plots the time complexity of normal dft, recursive dft and iterative fft.
    """
    # n is sizes of matrices
    # the values of this list can be changed ( conditions: n != 0 and n[i] <= n[j], for all i < j )
    n = [128, 256, 512, 1024, 2048, 4096]

    # count the time taken of each algorithm for the signal transformation with corresponding n

    elements = list()
    times = list()
    elements2 = list()
    times2 = list()
    elements3 = list()
    times3 = list()

    # i is the size from n used in each iteration
    for i in n:
        signal = np.random.rand(i)
        start = time.time()
        dftSignal = dft(i) @ signal
        end = time.time()
        elements.append(i)
        times.append(end - start)

        start = time.time()
        fftSignal = fft.fft(signal)
        end = time.time()
        elements2.append(i)
        times2.append(end - start)

        start = time.time()
        dftRecSignal = dft_recursive(signal)
        end = time.time()
        elements3.append(i)
        times3.append(end - start)

    plt.xlabel('Size of N')
    plt.ylabel('Time taken')
    plt.plot(elements, times, label='normal dft')
    plt.plot(elements3, times3, label='recursive dft')
    plt.plot(elements2, times2, label='iterative fft')

    plt.grid()
    plt.legend()
    plt.show()


def plotMatrix(N: int):
    """
    :param N: size input

    Plots the matrices of size N and 2N with colors. Density of matrices in the shown figure increases as n increases.
    Matrices in size N and 2N show the same pattern.
    """
    M = dft(N)
    M2 = dft(2 * N)

    plt.figure(figsize=(12, 10))

    # choose and change colormap as you want.
    # https://matplotlib.org/stable/tutorials/colors/colormaps.html
    colormap = 'nipy_spectral'

    plt.subplot(2, 2, 1)
    plt.title('$\mathrm{Re}(\mathrm{DFT}_N)$')
    plt.imshow(np.real(M), origin='lower', cmap=colormap, aspect='equal')
    plt.xlabel('Time index')
    plt.ylabel('Frequency index')
    plt.colorbar()

    plt.subplot(2, 2, 2)
    plt.title('$\mathrm{Im}(\mathrm{DFT}_N)$')
    plt.imshow(np.imag(M), origin='lower', cmap=colormap, aspect='equal')
    plt.xlabel('Time index')
    plt.ylabel('Frequency index')
    plt.colorbar()

    plt.subplot(2, 2, 3)
    plt.title('$\mathrm{Re}(\mathrm{DFT}_{2N})$')
    plt.imshow(np.real(M2), origin='lower', cmap=colormap, aspect='equal')
    plt.xlabel('Time index')
    plt.ylabel('Frequency index')
    plt.colorbar()

    plt.subplot(2, 2, 4)
    plt.title('$\mathrm{Im}(\mathrm{DFT}_{2N})$')
    plt.imshow(np.imag(M2), origin='lower', cmap=colormap, aspect='equal')
    plt.xlabel('Time index')
    plt.ylabel('Frequency index')
    plt.colorbar()

    plt.show()


if __name__ == '__main__':
    plotMatrix(16)
    plotMatrix(128)
    plotTime()

