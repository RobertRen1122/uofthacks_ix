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
        trace=trace[-100:]
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

        NN=4
        for theta in range(NN):
            if theta!=3:
                x_2=np.concatenate([x_2,np.cos(theta/NN*np.pi)*x])
                y_2=np.concatenate([y_2,Y])
                z_2=np.concatenate([z_2,-np.sin(theta/NN*np.pi)*x])
                x_2=np.concatenate([x_2,x])
                y_2=np.concatenate([y_2,np.cos(theta/NN*np.pi)*Y])
                z_2=np.concatenate([z_2,np.sin(theta/NN*np.pi)*Y])
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
        for i in range(NN-2):
            distances=np.concatenate([distances,
                                      trace_tp[2],trace_tp[2]])
        #print(distances.shape)
        #print(x_2.shape)

        result.append([x_2,y_2,z_2,distances])
    return result
        
if __name__ == "__main__":
    audio_path = "snake.wav"
    y, sampling_rate = librosa.load(audio_path, sr=8000)
    
    music_length = 30#len(y) / sampling_rate #in seconds
    update_interval = 0.03 #in seconds
    
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
    ax = plt.axes(projection='3d', xlim=(-1, 1),
                  ylim=(-1, 1), zlim=(-1,1))
    
    num = 10
    # Hide grid lines
    
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_axis_off()
    '''
    
    x_his_1 = np.linspace(-1, 10, num)
    x_his_2 = np.linspace(-10, 1, num)
    height = 16 # even
    z_his = np.ones(num)
    for i in range(len(z_his)):
        z_his[i] = height
    cm_b = plt.get_cmap("cividis")
    col_b = [cm_b(float(i) / len(x_his_1)) for i in range(num*2)]
    background = ax.bar(x_his_1, z_his, zs=1, zdir='y', alpha=0.5,
                        align='edge', bottom=-1*height/2, width=2/num, color = col_b[num:])
    background_1 = ax.bar(x_his_2, z_his, zs=-1, zdir='x', alpha=0.5, align='edge',
                          bottom=-1*height/2, width=2/num, color=col_b[:num])
    
    normal = np.array([1, -1, 1])
    point = np.array([5, -5, -5])
    d = -point.dot(normal)
    xx, yy = np.meshgrid(range(10), range(10))
    z = (-normal[0] * xx - normal[1] * yy - d) * 1. /normal[2]
    ax.plot_surface(xx, xx, z, cmap=plt.cm.YlGnBu_r, zorder=-10)
    '''
    r = 3
    u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
    x = r*np.cos(u) * np.sin(v)
    y = r*np.sin(u) * np.sin(v)
    z = r*np.cos(v)
    ax.plot_surface(x, y, z, cmap=plt.cm.YlGnBu_r, zorder=-10, alpha=0.1)


    #alphas=[0.1 for i in range(len(music_FFT[0][0]))]
    graph = ax.scatter(music_FFT[0][0],
                       music_FFT[0][1],
                       music_FFT[0][2], marker='o',zorder=10)
    angle=0
        
    def update(frame):
        global angle
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
        graph._alpha = [min(trace[3][i]*3, 1) for i in range(len(trace[3]))]
        graph._sizes = trace[3]*50

        cm = plt.get_cmap("viridis")
        col = [cm(float(i) / len(music_FFT[0][0])) for i in range(len(music_FFT[0][0]))]
        graph._facecolor3d = col
        graph._edgecolor3d = col

        ax.view_init(elev=32., azim=angle)
        angle += 16
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
    
