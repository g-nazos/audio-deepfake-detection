
import subprocess
from scipy.io.wavfile import read, write
import numpy as np
import tqdm
# function to use ffmpeg in python and get the detected silence.
def detect_silence(path,time):
    '''
    This function is a python wrapper to run the ffmpeg command in python and extranct the desired output
    time = silence time threshold
    returns = list of tuples with start and end point of silences
    '''
    command="ffmpeg -i "+path+" -af silencedetect=n=-23dB:d="+str(time)+" -f null -"
    out = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout, stderr = out.communicate()
    s=stdout.decode("utf-8")
    k=s.split('[silencedetect @')
    if len(k)==1:
        #print(stderr)
        return None
        
    start,end=[],[]
    for i in range(1,len(k)):
        x=k[i].split(']')[1]
        if i%2==0:
            x=x.split('|')[0]
            x=x.split(':')[1].strip()
            end.append(float(x))
        else:
            x=x.split(':')[1]
            x=x.split('size')[0]
            x=x.replace('\r','')
            x=x.replace('\n','').strip()
            start.append(float(x))
    return list(zip(start,end))

def remove_silence(file,sil,keep_sil,out_path):
    '''
    This function removes silence from the audio.
    
    Input: 
    file = Input audio file path
    sil = List of silence time slots that needs to be removed
    keep_sil = Time to keep as allowed silence after removing silence
    out_path = Output path of audio file
    
    returns:
    Non - silent patches and save the new audio in out path
    '''
    rate,aud=read(path)
    a=float(keep_sil)/2
    sil_updated=[(i[0]+a,i[1]-a) for i in sil]
    
    # convert the silence patch to non-sil patches
    non_sil=[]
    tmp=0
    ed=len(aud)/rate
    for i in range(len(sil_updated)):
        non_sil.append((tmp,sil_updated[i][0]))
        tmp=sil_updated[i][1]
    if sil_updated[-1][1]+a/2<ed:
        non_sil.append((sil_updated[-1][1],ed))
    if non_sil[0][0]==non_sil[0][1]:
        del non_sil[0]
    
    # cut the audio
    print('slicing starte')
    ans=[]
    ad=list(aud)
    for i in tqdm.tqdm(non_sil):
        ans=ans+ad[int(i[0]*rate):int(i[1]*rate)]
    #nm=path.split('/')[-1]
    write(out_path,rate,np.array(ans))
    return non_sil