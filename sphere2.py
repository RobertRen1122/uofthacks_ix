import time
import librosa
import matplotlib.pyplot as plt
import numpy as np
import vlc
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

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
            trace.append([centers[-1].real, centers[-1].imag,0])
        trace=np.array(trace)
    
        idx=[i*2 for i in range(N//2)]
        #idx.append(N-1)
        trace=trace[idx,:]

        #print(trace)
        trace_tp=np.transpose(trace)
        #result.append(trace_tp)
        
        x=trace_tp[0]
        Y=trace_tp[1]
        
        x_2=[]
        y_2=[]
        z_2=[]

        NN=10
        for theta in range(NN):
            for i in range(len(x)):
                x_2.append(np.cos(theta/NN/2*np.pi)*x[i])
                y_2.append(Y[i])
                z_2.append(-np.sin(theta/NN/2*np.pi)*x[i])
                x_2.append(x[i])
                y_2.append(np.cos(theta/NN/2*np.pi)*Y[i])
                z_2.append(np.sin(theta/NN/2*np.pi)*Y[i])
        result.append([x_2,y_2,z_2])
    return result
        

if __name__ == "__main__":
    audio_path = "Makaih Beats - Don't Make Assumptions.wav"
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

    fig = plt.figure()
    ax = plt.axes(projection='3d', xlim=(-0.8, 0.8),
                  ylim=(-0.8, 0.8), zlim=(-0.8,0.8))
    graph = ax.scatter(music_FFT[0][0],
                       music_FFT[0][1],
                       music_FFT[0][2], s=5, alpha=0.1)
    
    def update(frame):
        current_time = time.time()
        current_frame = ((current_time - music_play_start_time) //
                         update_interval)
        if current_frame == num_frames - 1:
            plt.close(fig)
            return #rects

        trace = music_FFT[int(current_frame)]
        graph._offsets3d = (trace[0],
                            trace[1],
                            trace[2])

        #time.sleep(update_interval)
        #print(time.time()-current_time)
        #return line,
    
    ani = FuncAnimation(fig, update, blit=False, interval=update_interval*1000,
                        frames=num_frames + 1, repeat=False)
    print("begin!")
    player.play()

    music_play_start_time = time.time()
    plt.show(block=False)
    plt.pause(music_length)
    
