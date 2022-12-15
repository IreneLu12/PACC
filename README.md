# PACC
**PACC: PERCEPTION AWARE CONGESTION CONTROL FOR REAL-TIME COMMUNICATION**

This repo contains the implementation code of PACC.  Please report any problem you faced or suggestion you have to improve the code.



## How to Use

### PVQS testing

To test the important component PVQS, you should have pytorch installed (GPU version is better) firstly. Then: 

```shell
git clone https://github.com/anonymity-icme2023/PACC.git
cd PVQS
python PVQS_test.py
```



### PACC testing

PACC is integrated in to [AlphaRTC](https://github.com/OpenNetLab/AlphaRTC) . To test it, the follow steps are required:

1. Compile AlphaRTC
1. Install [mahimahi](https://github.com/ravinet/mahimahi.git)



Because the above two steps are very complicated and tedious, we try to use docker to make testing easier. The detailed guidelines will be updated soon.



