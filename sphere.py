import time
import librosa
import matplotlib.pyplot as plt
import numpy as np
import vlc
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

def get_FFT(y,music_length,sampling_rate,update_interval):
    result=[]
    frame_interval = int(sampling_rate * update_interval)
    for k in range(int(music_length // update_interval)):
        N=frame_interval
        Z = np.fft.fft(y[frame_interval * k: frame_interval * (k + 1)], N)/N
        t = np.linspace(0, 2*np.pi, N)
        
        k_sorted = np.argsort(-np.abs(Z))  # these indices can be thought of as the frequencies
        Z = Z[k_sorted]

        # animate
        trace = []
        for n in range(N):
            centers = np.pad(np.cumsum(np.abs(Z) * np.exp(1j * ((k_sorted+1) * t[n]+np.angle(Z)))), [1, 0])
            trace.append([centers[-1].real, centers[-1].imag,
                          np.abs(centers[-1])])
        #print(trace)
    
        trace=np.array(trace)
        trace=trace[trace[:, 2].argsort()]
        trace=trace[-80:]
        #NN=2
        #idx=[i*NN for i in range(N//NN)]
        #idx.append(N-1)
        #trace=trace[idx,:]
        #print(trace)

        #print(trace)
        trace_tp=np.transpose(trace)
        #print(trace_tp.shape)
        #result.append(trace_tp)
        
        x=trace_tp[0]
        Y=trace_tp[1]
        
        x_2=[]
        y_2=[]
        z_2=[]

        NN=6
        for theta in range(NN):
            x_2=np.concatenate([x_2,np.cos(theta/NN/2*np.pi)*x])
            y_2=np.concatenate([y_2,Y])
            z_2=np.concatenate([z_2,-np.sin(theta/NN/2*np.pi)*x])
            x_2=np.concatenate([x_2,x])
            y_2=np.concatenate([y_2,np.cos(theta/NN/2*np.pi)*Y])
            z_2=np.concatenate([z_2,np.sin(theta/NN/2*np.pi)*Y])
            '''
            for i in range(len(x)):
                x_2.append(np.cos(theta/NN/2*np.pi)*x[i])
                y_2.append(Y[i])
                z_2.append(-np.sin(theta/NN/2*np.pi)*x[i])
                x_2.append(x[i])
                y_2.append(np.cos(theta/NN/2*np.pi)*Y[i])
                z_2.append(np.sin(theta/NN/2*np.pi)*Y[i])
            '''
        distances=np.concatenate([trace_tp[2],trace_tp[2]])
        for i in range(NN-1):
            distances=np.concatenate([distances,
                                      trace_tp[2],trace_tp[2]])
        #print(distances.shape)
        #print(x_2.shape)

        result.append([x_2,y_2,z_2,distances])
    return result
        
if __name__ == "__main__":
    audio_path = "snake.wav"
    y, sampling_rate = librosa.load(audio_path, sr=8000)
    
    music_length = 30 #len(y) / sampling_rate #in seconds
    update_interval = 0.05 #in seconds
    
    print(music_length, sampling_rate)
    player = vlc.MediaPlayer(audio_path)

    start=time.time()
    music_FFT = get_FFT(y=y, music_length=music_length,
                        sampling_rate=sampling_rate,
                        update_interval=update_interval)
    print(time.time()-start)
    
    num_frames = int(music_length // update_interval)
    music_play_start_time = 0

    fig = plt.figure()
    ax = plt.axes(projection='3d', xlim=(-0.8, 0.8),
                  ylim=(-0.8, 0.8), zlim=(-0.8,0.8))
    #alphas=[0.1 for i in range(len(music_FFT[0][0]))]
    graph = ax.scatter(music_FFT[0][0],
                       music_FFT[0][1],
                       music_FFT[0][2])
    
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
        graph._alpha = [min(trace[3][i], 1) for i in range(len(trace[3]))]
        graph._sizes = trace[3]*50
        
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
    
