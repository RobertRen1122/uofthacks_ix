import time
import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft
import vlc
from matplotlib.animation import FuncAnimation

CONST=1
def get_FFT(y,music_length,sampling_rate,update_interval):

    result=[]
    frame_interval = int(sampling_rate * update_interval)
    for k in range(int(music_length // update_interval)):
        N=64
        Z = np.fft.fft(y[frame_interval * k: frame_interval * (k + 1)], N)/N
        t = np.linspace(0, 2*np.pi*CONST, N)
        
        k_sorted = np.argsort(-np.abs(Z))  # these indices can be thought of as the frequencies
        Z = Z[k_sorted]
        
        # animate
        trace = []
        for n in range(len(Z)):
            centers = np.pad(np.cumsum(Z * np.exp(1j * (k_sorted+1)/CONST * t[n])), [1, 0])
            trace.append([centers[-1].real, centers[-1].imag])
        
        result.append(trace)
    return result
        
fig = plt.figure()
ax = plt.axes(xlim=(-0.5, 0.5), ylim=(-0.5, 0.5))
line, = ax.plot([], [], lw=2)
def init():
    #ax.clear()
    #ax.set(xlim=[-1, 1], ylim=[-1, 1])
    #ax.axis('off')
    line.set_data([], [])
    return line,
    

if __name__ == "__main__":
    audio_path = "barradeen-bedtime-after-a-coffee.wav"
    y, sampling_rate = librosa.load(audio_path, sr=None)
    
    music_length = len(y) / sampling_rate #in seconds
    update_interval = 0.1 #in seconds
    
    print(music_length)
    player = vlc.MediaPlayer(audio_path)
    
    music_FFT = get_FFT(y=y, music_length=music_length,
                        sampling_rate=sampling_rate,
                        update_interval=update_interval)
    print("done")
    
    num_frames = int(music_length // update_interval)
    music_play_start_time = 0
    
    def update(frame):
        current_time = time.time()
        current_frame = ((current_time - music_play_start_time) //
                         update_interval)
        if current_frame == num_frames - 1:
            plt.close(fig)
            return #rects

        trace = music_FFT[int(current_frame)]
        line.set_data(*zip(*trace))

        time.sleep(update_interval)
        return line,
    
    ani = FuncAnimation(fig, update, init_func=init, blit=True, interval=0,
                        frames=num_frames + 1, repeat=False)
    print("begin!")
    player.play()

    music_play_start_time = time.time()
    plt.show(block=False)
    plt.pause(music_length)
    
