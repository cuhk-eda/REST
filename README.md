# REST

REST is a reinforcement learning framework for rectilinear Steiner minimum tree (RSMT) construction.
Read our paper for more details:

* Jinwei Liu, Gengjie Chen, and Evangeline FY Young. "[REST: Constructing Rectilinear Steiner Minimum Tree via Reinforcement Learning.](https://ieeexplore.ieee.org/document/9586209)" 2021 58th ACM/IEEE Design Automation Conference (DAC). IEEE, 2021.

## Dependencies
* Python 3.6+
* PyTorch 1.10.0+
* GeoSteiner 5.1 (included)

## Training
Start a new training process for degree 20 by

~~~bash
python3 train.py --degree 20
~~~
<br>

## Testing
1. Test the trained model using randomly generated data set by

~~~bash
python3 test.py --degree 20
~~~
<br>

2. Or use the trained parameters included with this repository

~~~bash
python3 test.py --degree 20 --experiment DAC21
~~~
<br>

3. As mentioned in the paper, the percentage error can be further reduced by using 
multiple transformations of the input point set for inference. Inference using 
all eight transformations by

~~~bash
python3 test.py --degree 20 --experiment DAC21 --transformation 8
~~~
<br>

4. Lastly, if you want to test the same data set as in the paper

~~~bash
python3 test.py --degree 20 --test_data test_set/test20.txt
~~~
<br>

## Results
Using only one of the transformations for inference

![rest_20_t1.png](/images/rest_20_t1.png)

Using all eight transformations for inference

![rest_20_t8.png](/images/rest_20_t8.png)

## License
READ THIS LICENSE AGREEMENT CAREFULLY BEFORE USING THIS PRODUCT. BY USING THIS 
PRODUCT YOU INDICATE YOUR ACCEPTANCE OF THE TERMS OF THE FOLLOWING AGREEMENT. 
THESE TERMS APPLY TO YOU AND ANY SUBSEQUENT LICENSEE OF THIS PRODUCT.

License Agreement for REST

Copyright (c) 2022, The Chinese University of Hong Kong
All rights reserved.

CU-SD LICENSE (adapted from the original BSD license) Redistribution of the any 
code, with or without modification, are permitted provided that the conditions 
below are met. 

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name nor trademark of the copyright holder or the author may be 
   used to endorse or promote products derived from this software without 
   specific prior written permission.

4. Users are entirely responsible, to the exclusion of the author, for 
   compliance with (a) regulations set by owners or administrators of employed 
   equipment, (b) licensing terms of any other software, and (c) local, 
   national, and international regulations regarding use, including those 
   regarding import, export, and use of encryption software.

THIS FREE SOFTWARE IS PROVIDED BY THE AUTHOR "AS IS" AND ANY EXPRESS OR IMPLIED 
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF 
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT 
SHALL THE AUTHOR OR ANY CONTRIBUTOR BE LIABLE FOR ANY DIRECT, INDIRECT, 
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT 
LIMITED TO, EFFECTS OF UNAUTHORIZED OR MALICIOUS NETWORK ACCESS; PROCUREMENT OF 
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
POSSIBILITY OF SUCH DAMAGE.


