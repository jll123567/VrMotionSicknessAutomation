м
џг
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 

BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

ж
Conv3D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)(0""
paddingstring:
SAMEVALID"0
data_formatstringNDHWC:
NDHWCNCDHW"!
	dilations	list(int)	

$
DisableCopyOnRead
resource
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
ћ
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%Зб8"&
exponential_avg_factorfloat%  ?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( ""
Ttype:
2	"
Tidxtype0:
2	

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
С
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring Ј
@
StaticRegexFullMatch	
input

output
"
patternstring
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
А
VarHandleOp
resource"
	containerstring "
shared_namestring "

debug_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.15.02v2.15.0-2-g0b15fdfcb3f8И
v
countVarHandleOp*
_output_shapes
: *

debug_namecount/*
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
v
totalVarHandleOp*
_output_shapes
: *

debug_nametotal/*
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
|
count_1VarHandleOp*
_output_shapes
: *

debug_name
count_1/*
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
|
total_1VarHandleOp*
_output_shapes
: *

debug_name
total_1/*
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0

learning_rateVarHandleOp*
_output_shapes
: *

debug_namelearning_rate/*
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0

	iterationVarHandleOp*
_output_shapes
: *

debug_name
iteration/*
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	

dense_5/biasVarHandleOp*
_output_shapes
: *

debug_namedense_5/bias/*
dtype0*
shape:*
shared_namedense_5/bias
i
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
_output_shapes
:*
dtype0

dense_5/kernelVarHandleOp*
_output_shapes
: *

debug_namedense_5/kernel/*
dtype0*
shape
:*
shared_namedense_5/kernel
q
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel*
_output_shapes

:*
dtype0

dense_4/biasVarHandleOp*
_output_shapes
: *

debug_namedense_4/bias/*
dtype0*
shape:*
shared_namedense_4/bias
i
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes
:*
dtype0

dense_4/kernelVarHandleOp*
_output_shapes
: *

debug_namedense_4/kernel/*
dtype0*
shape
:*
shared_namedense_4/kernel
q
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*
_output_shapes

:*
dtype0

dense_3/biasVarHandleOp*
_output_shapes
: *

debug_namedense_3/bias/*
dtype0*
shape:*
shared_namedense_3/bias
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes
:*
dtype0

dense_3/kernelVarHandleOp*
_output_shapes
: *

debug_namedense_3/kernel/*
dtype0*
shape
: *
shared_namedense_3/kernel
q
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes

: *
dtype0

dense_2/biasVarHandleOp*
_output_shapes
: *

debug_namedense_2/bias/*
dtype0*
shape: *
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
: *
dtype0

dense_2/kernelVarHandleOp*
_output_shapes
: *

debug_namedense_2/kernel/*
dtype0*
shape
:
 *
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

:
 *
dtype0

dense_1/biasVarHandleOp*
_output_shapes
: *

debug_namedense_1/bias/*
dtype0*
shape:*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:*
dtype0

dense_1/kernelVarHandleOp*
_output_shapes
: *

debug_namedense_1/kernel/*
dtype0*
shape
:@*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:@*
dtype0


dense/biasVarHandleOp*
_output_shapes
: *

debug_namedense/bias/*
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0

dense/kernelVarHandleOp*
_output_shapes
: *

debug_namedense/kernel/*
dtype0*
shape
:@*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:@*
dtype0
к
%batch_normalization_5/moving_varianceVarHandleOp*
_output_shapes
: *6

debug_name(&batch_normalization_5/moving_variance/*
dtype0*
shape:@*6
shared_name'%batch_normalization_5/moving_variance

9batch_normalization_5/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_5/moving_variance*
_output_shapes
:@*
dtype0
Ю
!batch_normalization_5/moving_meanVarHandleOp*
_output_shapes
: *2

debug_name$"batch_normalization_5/moving_mean/*
dtype0*
shape:@*2
shared_name#!batch_normalization_5/moving_mean

5batch_normalization_5/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_5/moving_mean*
_output_shapes
:@*
dtype0
Й
batch_normalization_5/betaVarHandleOp*
_output_shapes
: *+

debug_namebatch_normalization_5/beta/*
dtype0*
shape:@*+
shared_namebatch_normalization_5/beta

.batch_normalization_5/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_5/beta*
_output_shapes
:@*
dtype0
М
batch_normalization_5/gammaVarHandleOp*
_output_shapes
: *,

debug_namebatch_normalization_5/gamma/*
dtype0*
shape:@*,
shared_namebatch_normalization_5/gamma

/batch_normalization_5/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_5/gamma*
_output_shapes
:@*
dtype0
к
%batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *6

debug_name(&batch_normalization_2/moving_variance/*
dtype0*
shape:@*6
shared_name'%batch_normalization_2/moving_variance

9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
_output_shapes
:@*
dtype0
Ю
!batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *2

debug_name$"batch_normalization_2/moving_mean/*
dtype0*
shape:@*2
shared_name#!batch_normalization_2/moving_mean

5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
_output_shapes
:@*
dtype0
Й
batch_normalization_2/betaVarHandleOp*
_output_shapes
: *+

debug_namebatch_normalization_2/beta/*
dtype0*
shape:@*+
shared_namebatch_normalization_2/beta

.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
_output_shapes
:@*
dtype0
М
batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *,

debug_namebatch_normalization_2/gamma/*
dtype0*
shape:@*,
shared_namebatch_normalization_2/gamma

/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
_output_shapes
:@*
dtype0

conv3d_2/biasVarHandleOp*
_output_shapes
: *

debug_nameconv3d_2/bias/*
dtype0*
shape:@*
shared_nameconv3d_2/bias
k
!conv3d_2/bias/Read/ReadVariableOpReadVariableOpconv3d_2/bias*
_output_shapes
:@*
dtype0
Ј
conv3d_2/kernelVarHandleOp*
_output_shapes
: * 

debug_nameconv3d_2/kernel/*
dtype0*
shape:@@* 
shared_nameconv3d_2/kernel

#conv3d_2/kernel/Read/ReadVariableOpReadVariableOpconv3d_2/kernel**
_output_shapes
:@@*
dtype0

conv1d_2/biasVarHandleOp*
_output_shapes
: *

debug_nameconv1d_2/bias/*
dtype0*
shape:@*
shared_nameconv1d_2/bias
k
!conv1d_2/bias/Read/ReadVariableOpReadVariableOpconv1d_2/bias*
_output_shapes
:@*
dtype0
 
conv1d_2/kernelVarHandleOp*
_output_shapes
: * 

debug_nameconv1d_2/kernel/*
dtype0*
shape:@@* 
shared_nameconv1d_2/kernel
w
#conv1d_2/kernel/Read/ReadVariableOpReadVariableOpconv1d_2/kernel*"
_output_shapes
:@@*
dtype0
к
%batch_normalization_4/moving_varianceVarHandleOp*
_output_shapes
: *6

debug_name(&batch_normalization_4/moving_variance/*
dtype0*
shape:@*6
shared_name'%batch_normalization_4/moving_variance

9batch_normalization_4/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_4/moving_variance*
_output_shapes
:@*
dtype0
Ю
!batch_normalization_4/moving_meanVarHandleOp*
_output_shapes
: *2

debug_name$"batch_normalization_4/moving_mean/*
dtype0*
shape:@*2
shared_name#!batch_normalization_4/moving_mean

5batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_4/moving_mean*
_output_shapes
:@*
dtype0
Й
batch_normalization_4/betaVarHandleOp*
_output_shapes
: *+

debug_namebatch_normalization_4/beta/*
dtype0*
shape:@*+
shared_namebatch_normalization_4/beta

.batch_normalization_4/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_4/beta*
_output_shapes
:@*
dtype0
М
batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *,

debug_namebatch_normalization_4/gamma/*
dtype0*
shape:@*,
shared_namebatch_normalization_4/gamma

/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_4/gamma*
_output_shapes
:@*
dtype0
к
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *6

debug_name(&batch_normalization_1/moving_variance/*
dtype0*
shape:@*6
shared_name'%batch_normalization_1/moving_variance

9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes
:@*
dtype0
Ю
!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *2

debug_name$"batch_normalization_1/moving_mean/*
dtype0*
shape:@*2
shared_name#!batch_normalization_1/moving_mean

5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes
:@*
dtype0
Й
batch_normalization_1/betaVarHandleOp*
_output_shapes
: *+

debug_namebatch_normalization_1/beta/*
dtype0*
shape:@*+
shared_namebatch_normalization_1/beta

.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes
:@*
dtype0
М
batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *,

debug_namebatch_normalization_1/gamma/*
dtype0*
shape:@*,
shared_namebatch_normalization_1/gamma

/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes
:@*
dtype0

conv3d_1/biasVarHandleOp*
_output_shapes
: *

debug_nameconv3d_1/bias/*
dtype0*
shape:@*
shared_nameconv3d_1/bias
k
!conv3d_1/bias/Read/ReadVariableOpReadVariableOpconv3d_1/bias*
_output_shapes
:@*
dtype0
Ј
conv3d_1/kernelVarHandleOp*
_output_shapes
: * 

debug_nameconv3d_1/kernel/*
dtype0*
shape:@@* 
shared_nameconv3d_1/kernel

#conv3d_1/kernel/Read/ReadVariableOpReadVariableOpconv3d_1/kernel**
_output_shapes
:@@*
dtype0

conv1d_1/biasVarHandleOp*
_output_shapes
: *

debug_nameconv1d_1/bias/*
dtype0*
shape:@*
shared_nameconv1d_1/bias
k
!conv1d_1/bias/Read/ReadVariableOpReadVariableOpconv1d_1/bias*
_output_shapes
:@*
dtype0
 
conv1d_1/kernelVarHandleOp*
_output_shapes
: * 

debug_nameconv1d_1/kernel/*
dtype0*
shape:@@* 
shared_nameconv1d_1/kernel
w
#conv1d_1/kernel/Read/ReadVariableOpReadVariableOpconv1d_1/kernel*"
_output_shapes
:@@*
dtype0
к
%batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *6

debug_name(&batch_normalization_3/moving_variance/*
dtype0*
shape:@*6
shared_name'%batch_normalization_3/moving_variance

9batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_3/moving_variance*
_output_shapes
:@*
dtype0
Ю
!batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *2

debug_name$"batch_normalization_3/moving_mean/*
dtype0*
shape:@*2
shared_name#!batch_normalization_3/moving_mean

5batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*
_output_shapes
:@*
dtype0
Й
batch_normalization_3/betaVarHandleOp*
_output_shapes
: *+

debug_namebatch_normalization_3/beta/*
dtype0*
shape:@*+
shared_namebatch_normalization_3/beta

.batch_normalization_3/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_3/beta*
_output_shapes
:@*
dtype0
М
batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *,

debug_namebatch_normalization_3/gamma/*
dtype0*
shape:@*,
shared_namebatch_normalization_3/gamma

/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_3/gamma*
_output_shapes
:@*
dtype0
д
#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *4

debug_name&$batch_normalization/moving_variance/*
dtype0*
shape:@*4
shared_name%#batch_normalization/moving_variance

7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes
:@*
dtype0
Ш
batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *0

debug_name" batch_normalization/moving_mean/*
dtype0*
shape:@*0
shared_name!batch_normalization/moving_mean

3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
:@*
dtype0
Г
batch_normalization/betaVarHandleOp*
_output_shapes
: *)

debug_namebatch_normalization/beta/*
dtype0*
shape:@*)
shared_namebatch_normalization/beta

,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes
:@*
dtype0
Ж
batch_normalization/gammaVarHandleOp*
_output_shapes
: **

debug_namebatch_normalization/gamma/*
dtype0*
shape:@**
shared_namebatch_normalization/gamma

-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes
:@*
dtype0

conv3d/biasVarHandleOp*
_output_shapes
: *

debug_nameconv3d/bias/*
dtype0*
shape:@*
shared_nameconv3d/bias
g
conv3d/bias/Read/ReadVariableOpReadVariableOpconv3d/bias*
_output_shapes
:@*
dtype0
Ђ
conv3d/kernelVarHandleOp*
_output_shapes
: *

debug_nameconv3d/kernel/*
dtype0*
shape:@*
shared_nameconv3d/kernel
{
!conv3d/kernel/Read/ReadVariableOpReadVariableOpconv3d/kernel**
_output_shapes
:@*
dtype0

conv1d/biasVarHandleOp*
_output_shapes
: *

debug_nameconv1d/bias/*
dtype0*
shape:@*
shared_nameconv1d/bias
g
conv1d/bias/Read/ReadVariableOpReadVariableOpconv1d/bias*
_output_shapes
:@*
dtype0

conv1d/kernelVarHandleOp*
_output_shapes
: *

debug_nameconv1d/kernel/*
dtype0*
shape:t@*
shared_nameconv1d/kernel
s
!conv1d/kernel/Read/ReadVariableOpReadVariableOpconv1d/kernel*"
_output_shapes
:t@*
dtype0

serving_default_input_1Placeholder*+
_output_shapes
:џџџџџџџџџdt*
dtype0* 
shape:џџџџџџџџџdt

serving_default_input_2Placeholder*5
_output_shapes#
!:џџџџџџџџџd*
dtype0**
shape!:џџџџџџџџџd
Ё
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1serving_default_input_2conv3d/kernelconv3d/biasconv1d/kernelconv1d/biasbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_variance#batch_normalization/moving_variancebatch_normalization/gammabatch_normalization/moving_meanbatch_normalization/betaconv3d_1/kernelconv3d_1/biasconv1d_1/kernelconv1d_1/biasbatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_variance%batch_normalization_1/moving_variancebatch_normalization_1/gamma!batch_normalization_1/moving_meanbatch_normalization_1/betaconv3d_2/kernelconv3d_2/biasconv1d_2/kernelconv1d_2/biasbatch_normalization_5/gammabatch_normalization_5/beta!batch_normalization_5/moving_mean%batch_normalization_5/moving_variance%batch_normalization_2/moving_variancebatch_normalization_2/gamma!batch_normalization_2/moving_meanbatch_normalization_2/betadense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/biasdense_4/kerneldense_4/biasdense_5/kerneldense_5/bias*=
Tin6
422*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*R
_read_only_resource_inputs4
20	
 !"#$%&'()*+,-./01*2
config_proto" 

CPU

GPU2*0,1J 8 *+
f&R$
"__inference_signature_wrapper_2201

NoOpNoOp
ЫО
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*О
valueњНBіН BюН
Д
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer_with_weights-6
layer-10
layer_with_weights-7
layer-11
layer-12
layer-13
layer_with_weights-8
layer-14
layer_with_weights-9
layer-15
layer_with_weights-10
layer-16
layer_with_weights-11
layer-17
layer-18
layer-19
layer-20
layer-21
layer_with_weights-12
layer-22
layer_with_weights-13
layer-23
layer-24
layer_with_weights-14
layer-25
layer_with_weights-15
layer-26
layer_with_weights-16
layer-27
layer_with_weights-17
layer-28
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses
$_default_save_signature
%	optimizer
&
signatures*
* 
* 
Ш
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses

-kernel
.bias
 /_jit_compiled_convolution_op*
Ш
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses

6kernel
7bias
 8_jit_compiled_convolution_op*
е
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses
?axis
	@gamma
Abeta
Bmoving_mean
Cmoving_variance*
е
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
H__call__
*I&call_and_return_all_conditional_losses
Jaxis
	Kgamma
Lbeta
Mmoving_mean
Nmoving_variance*

O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses* 

U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses* 
Ш
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses

akernel
bbias
 c_jit_compiled_convolution_op*
Ш
d	variables
etrainable_variables
fregularization_losses
g	keras_api
h__call__
*i&call_and_return_all_conditional_losses

jkernel
kbias
 l_jit_compiled_convolution_op*
е
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
q__call__
*r&call_and_return_all_conditional_losses
saxis
	tgamma
ubeta
vmoving_mean
wmoving_variance*
и
x	variables
ytrainable_variables
zregularization_losses
{	keras_api
|__call__
*}&call_and_return_all_conditional_losses
~axis
	gamma
	beta
moving_mean
moving_variance*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
б
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias
!_jit_compiled_convolution_op*
б
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias
! _jit_compiled_convolution_op*
р
Ё	variables
Ђtrainable_variables
Ѓregularization_losses
Є	keras_api
Ѕ__call__
+І&call_and_return_all_conditional_losses
	Їaxis

Јgamma
	Љbeta
Њmoving_mean
Ћmoving_variance*
р
Ќ	variables
­trainable_variables
Ўregularization_losses
Џ	keras_api
А__call__
+Б&call_and_return_all_conditional_losses
	Вaxis

Гgamma
	Дbeta
Еmoving_mean
Жmoving_variance*

З	variables
Иtrainable_variables
Йregularization_losses
К	keras_api
Л__call__
+М&call_and_return_all_conditional_losses* 

Н	variables
Оtrainable_variables
Пregularization_losses
Р	keras_api
С__call__
+Т&call_and_return_all_conditional_losses* 

У	variables
Фtrainable_variables
Хregularization_losses
Ц	keras_api
Ч__call__
+Ш&call_and_return_all_conditional_losses* 

Щ	variables
Ъtrainable_variables
Ыregularization_losses
Ь	keras_api
Э__call__
+Ю&call_and_return_all_conditional_losses* 
Ў
Я	variables
аtrainable_variables
бregularization_losses
в	keras_api
г__call__
+д&call_and_return_all_conditional_losses
еkernel
	жbias*
Ў
з	variables
иtrainable_variables
йregularization_losses
к	keras_api
л__call__
+м&call_and_return_all_conditional_losses
нkernel
	оbias*

п	variables
рtrainable_variables
сregularization_losses
т	keras_api
у__call__
+ф&call_and_return_all_conditional_losses* 
Ў
х	variables
цtrainable_variables
чregularization_losses
ш	keras_api
щ__call__
+ъ&call_and_return_all_conditional_losses
ыkernel
	ьbias*
Ў
э	variables
юtrainable_variables
яregularization_losses
№	keras_api
ё__call__
+ђ&call_and_return_all_conditional_losses
ѓkernel
	єbias*
Ў
ѕ	variables
іtrainable_variables
їregularization_losses
ј	keras_api
љ__call__
+њ&call_and_return_all_conditional_losses
ћkernel
	ќbias*
Ў
§	variables
ўtrainable_variables
џregularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias*

-0
.1
62
73
@4
A5
B6
C7
K8
L9
M10
N11
a12
b13
j14
k15
t16
u17
v18
w19
20
21
22
23
24
25
26
27
Ј28
Љ29
Њ30
Ћ31
Г32
Д33
Е34
Ж35
е36
ж37
н38
о39
ы40
ь41
ѓ42
є43
ћ44
ќ45
46
47*
Џ
-0
.1
62
73
@4
A5
K6
L7
a8
b9
j10
k11
t12
u13
14
15
16
17
18
19
Ј20
Љ21
Г22
Д23
е24
ж25
н26
о27
ы28
ь29
ѓ30
є31
ћ32
ќ33
34
35*
* 
Е
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
$_default_save_signature
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses*

trace_0
trace_1* 

trace_0
trace_1* 
* 
S

_variables
_iterations
_learning_rate
_update_step_xla*

serving_default* 

-0
.1*

-0
.1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses*

trace_0* 

trace_0* 
]W
VARIABLE_VALUEconv1d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEconv1d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

60
71*

60
71*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses*

trace_0* 

 trace_0* 
]W
VARIABLE_VALUEconv3d/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEconv3d/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
@0
A1
B2
C3*

@0
A1*
* 

Ёnon_trainable_variables
Ђlayers
Ѓmetrics
 Єlayer_regularization_losses
Ѕlayer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses*

Іtrace_0
Їtrace_1* 

Јtrace_0
Љtrace_1* 
* 
hb
VARIABLE_VALUEbatch_normalization/gamma5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEbatch_normalization/beta4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEbatch_normalization/moving_mean;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE#batch_normalization/moving_variance?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
K0
L1
M2
N3*

K0
L1*
* 

Њnon_trainable_variables
Ћlayers
Ќmetrics
 ­layer_regularization_losses
Ўlayer_metrics
D	variables
Etrainable_variables
Fregularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses*

Џtrace_0
Аtrace_1* 

Бtrace_0
Вtrace_1* 
* 
jd
VARIABLE_VALUEbatch_normalization_3/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEbatch_normalization_3/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE!batch_normalization_3/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE%batch_normalization_3/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

Гnon_trainable_variables
Дlayers
Еmetrics
 Жlayer_regularization_losses
Зlayer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses* 

Иtrace_0* 

Йtrace_0* 
* 
* 
* 

Кnon_trainable_variables
Лlayers
Мmetrics
 Нlayer_regularization_losses
Оlayer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses* 

Пtrace_0* 

Рtrace_0* 

a0
b1*

a0
b1*
* 

Сnon_trainable_variables
Тlayers
Уmetrics
 Фlayer_regularization_losses
Хlayer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses*

Цtrace_0* 

Чtrace_0* 
_Y
VARIABLE_VALUEconv1d_1/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv1d_1/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

j0
k1*

j0
k1*
* 

Шnon_trainable_variables
Щlayers
Ъmetrics
 Ыlayer_regularization_losses
Ьlayer_metrics
d	variables
etrainable_variables
fregularization_losses
h__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses*

Эtrace_0* 

Юtrace_0* 
_Y
VARIABLE_VALUEconv3d_1/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv3d_1/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
t0
u1
v2
w3*

t0
u1*
* 

Яnon_trainable_variables
аlayers
бmetrics
 вlayer_regularization_losses
гlayer_metrics
m	variables
ntrainable_variables
oregularization_losses
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses*

дtrace_0
еtrace_1* 

жtrace_0
зtrace_1* 
* 
jd
VARIABLE_VALUEbatch_normalization_1/gamma5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEbatch_normalization_1/beta4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE!batch_normalization_1/moving_mean;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE%batch_normalization_1/moving_variance?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
#
0
1
2
3*

0
1*
* 

иnon_trainable_variables
йlayers
кmetrics
 лlayer_regularization_losses
мlayer_metrics
x	variables
ytrainable_variables
zregularization_losses
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses*

нtrace_0
оtrace_1* 

пtrace_0
рtrace_1* 
* 
jd
VARIABLE_VALUEbatch_normalization_4/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEbatch_normalization_4/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE!batch_normalization_4/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE%batch_normalization_4/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

сnon_trainable_variables
тlayers
уmetrics
 фlayer_regularization_losses
хlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 

цtrace_0* 

чtrace_0* 
* 
* 
* 

шnon_trainable_variables
щlayers
ъmetrics
 ыlayer_regularization_losses
ьlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 

эtrace_0* 

юtrace_0* 

0
1*

0
1*
* 

яnon_trainable_variables
№layers
ёmetrics
 ђlayer_regularization_losses
ѓlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

єtrace_0* 

ѕtrace_0* 
_Y
VARIABLE_VALUEconv1d_2/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv1d_2/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1*

0
1*
* 

іnon_trainable_variables
їlayers
јmetrics
 љlayer_regularization_losses
њlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

ћtrace_0* 

ќtrace_0* 
_Y
VARIABLE_VALUEconv3d_2/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv3d_2/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
$
Ј0
Љ1
Њ2
Ћ3*

Ј0
Љ1*
* 

§non_trainable_variables
ўlayers
џmetrics
 layer_regularization_losses
layer_metrics
Ё	variables
Ђtrainable_variables
Ѓregularization_losses
Ѕ__call__
+І&call_and_return_all_conditional_losses
'І"call_and_return_conditional_losses*

trace_0
trace_1* 

trace_0
trace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_2/gamma6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_2/beta5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE!batch_normalization_2/moving_mean<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE%batch_normalization_2/moving_variance@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
$
Г0
Д1
Е2
Ж3*

Г0
Д1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ќ	variables
­trainable_variables
Ўregularization_losses
А__call__
+Б&call_and_return_all_conditional_losses
'Б"call_and_return_conditional_losses*

trace_0
trace_1* 

trace_0
trace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_5/gamma6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_5/beta5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE!batch_normalization_5/moving_mean<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE%batch_normalization_5/moving_variance@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
З	variables
Иtrainable_variables
Йregularization_losses
Л__call__
+М&call_and_return_all_conditional_losses
'М"call_and_return_conditional_losses* 

trace_0* 

trace_0* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Н	variables
Оtrainable_variables
Пregularization_losses
С__call__
+Т&call_and_return_all_conditional_losses
'Т"call_and_return_conditional_losses* 

trace_0* 

trace_0* 
* 
* 
* 

non_trainable_variables
layers
metrics
  layer_regularization_losses
Ёlayer_metrics
У	variables
Фtrainable_variables
Хregularization_losses
Ч__call__
+Ш&call_and_return_all_conditional_losses
'Ш"call_and_return_conditional_losses* 

Ђtrace_0* 

Ѓtrace_0* 
* 
* 
* 

Єnon_trainable_variables
Ѕlayers
Іmetrics
 Їlayer_regularization_losses
Јlayer_metrics
Щ	variables
Ъtrainable_variables
Ыregularization_losses
Э__call__
+Ю&call_and_return_all_conditional_losses
'Ю"call_and_return_conditional_losses* 

Љtrace_0* 

Њtrace_0* 

е0
ж1*

е0
ж1*
* 

Ћnon_trainable_variables
Ќlayers
­metrics
 Ўlayer_regularization_losses
Џlayer_metrics
Я	variables
аtrainable_variables
бregularization_losses
г__call__
+д&call_and_return_all_conditional_losses
'д"call_and_return_conditional_losses*

Аtrace_0* 

Бtrace_0* 
]W
VARIABLE_VALUEdense/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUE
dense/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE*

н0
о1*

н0
о1*
* 

Вnon_trainable_variables
Гlayers
Дmetrics
 Еlayer_regularization_losses
Жlayer_metrics
з	variables
иtrainable_variables
йregularization_losses
л__call__
+м&call_and_return_all_conditional_losses
'м"call_and_return_conditional_losses*

Зtrace_0* 

Иtrace_0* 
_Y
VARIABLE_VALUEdense_1/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_1/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

Йnon_trainable_variables
Кlayers
Лmetrics
 Мlayer_regularization_losses
Нlayer_metrics
п	variables
рtrainable_variables
сregularization_losses
у__call__
+ф&call_and_return_all_conditional_losses
'ф"call_and_return_conditional_losses* 

Оtrace_0* 

Пtrace_0* 

ы0
ь1*

ы0
ь1*
* 

Рnon_trainable_variables
Сlayers
Тmetrics
 Уlayer_regularization_losses
Фlayer_metrics
х	variables
цtrainable_variables
чregularization_losses
щ__call__
+ъ&call_and_return_all_conditional_losses
'ъ"call_and_return_conditional_losses*

Хtrace_0* 

Цtrace_0* 
_Y
VARIABLE_VALUEdense_2/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_2/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE*

ѓ0
є1*

ѓ0
є1*
* 

Чnon_trainable_variables
Шlayers
Щmetrics
 Ъlayer_regularization_losses
Ыlayer_metrics
э	variables
юtrainable_variables
яregularization_losses
ё__call__
+ђ&call_and_return_all_conditional_losses
'ђ"call_and_return_conditional_losses*

Ьtrace_0* 

Эtrace_0* 
_Y
VARIABLE_VALUEdense_3/kernel7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_3/bias5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUE*

ћ0
ќ1*

ћ0
ќ1*
* 

Юnon_trainable_variables
Яlayers
аmetrics
 бlayer_regularization_losses
вlayer_metrics
ѕ	variables
іtrainable_variables
їregularization_losses
љ__call__
+њ&call_and_return_all_conditional_losses
'њ"call_and_return_conditional_losses*

гtrace_0* 

дtrace_0* 
_Y
VARIABLE_VALUEdense_4/kernel7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_4/bias5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

еnon_trainable_variables
жlayers
зmetrics
 иlayer_regularization_losses
йlayer_metrics
§	variables
ўtrainable_variables
џregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

кtrace_0* 

лtrace_0* 
_Y
VARIABLE_VALUEdense_5/kernel7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_5/bias5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUE*
`
B0
C1
M2
N3
v4
w5
6
7
Њ8
Ћ9
Е10
Ж11*
т
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28*

м0
н1*
* 
* 
* 
* 
* 
* 

0*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

B0
C1*
* 
* 
* 
* 
* 
* 
* 
* 

M0
N1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

v0
w1*
* 
* 
* 
* 
* 
* 
* 
* 

0
1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

Њ0
Ћ1*
* 
* 
* 
* 
* 
* 
* 
* 

Е0
Ж1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
о	variables
п	keras_api

рtotal

сcount*
M
т	variables
у	keras_api

фtotal

хcount
ц
_fn_kwargs*

р0
с1*

о	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

ф0
х1*

т	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameconv1d/kernelconv1d/biasconv3d/kernelconv3d/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variancebatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_varianceconv1d_1/kernelconv1d_1/biasconv3d_1/kernelconv3d_1/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_variancebatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_varianceconv1d_2/kernelconv1d_2/biasconv3d_2/kernelconv3d_2/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_variancebatch_normalization_5/gammabatch_normalization_5/beta!batch_normalization_5/moving_mean%batch_normalization_5/moving_variancedense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/biasdense_4/kerneldense_4/biasdense_5/kerneldense_5/bias	iterationlearning_ratetotal_1count_1totalcountConst*C
Tin<
:28*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *&
f!R
__inference__traced_save_3315
ћ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv1d/kernelconv1d/biasconv3d/kernelconv3d/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variancebatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_varianceconv1d_1/kernelconv1d_1/biasconv3d_1/kernelconv3d_1/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_variancebatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_varianceconv1d_2/kernelconv1d_2/biasconv3d_2/kernelconv3d_2/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_variancebatch_normalization_5/gammabatch_normalization_5/beta!batch_normalization_5/moving_mean%batch_normalization_5/moving_variancedense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/biasdense_4/kerneldense_4/biasdense_5/kerneldense_5/bias	iterationlearning_ratetotal_1count_1totalcount*B
Tin;
927*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *)
f$R"
 __inference__traced_restore_3486ЂБ
щ

&__inference_dense_2_layer_call_fn_2900

inputs
unknown:
 
	unknown_0: 
identityЂStatefulPartitionedCallл
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_1567o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ
: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ

 
_user_specified_nameinputs:$ 

_user_specified_name2894:$ 

_user_specified_name2896
О
q
E__inference_concatenate_layer_call_and_return_conditional_losses_2891
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :w
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ
W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџ:џџџџџџџџџ:Q M
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_0:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_1
е
]
A__inference_re_lu_2_layer_call_and_return_conditional_losses_1514

inputs
identityJ
ReluReluinputs*
T0*+
_output_shapes
:џџџџџџџџџd@^
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:џџџџџџџџџd@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџd@:S O
+
_output_shapes
:џџџџџџџџџd@
 
_user_specified_nameinputs
ў

џ
B__inference_conv3d_1_layer_call_and_return_conditional_losses_2449

inputs<
conv3d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:@@*
dtype0 
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:џџџџџџџџџd@*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:џџџџџџџџџd@m
IdentityIdentityBiasAdd:output:0^NoOp*
T0*5
_output_shapes#
!:џџџџџџџџџd@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:џџџџџџџџџd@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:] Y
5
_output_shapes#
!:џџџџџџџџџd@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
џ

'__inference_conv1d_2_layer_call_fn_2620

inputs
unknown:@@
	unknown_0:@
identityЂStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџd@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *K
fFRD
B__inference_conv1d_2_layer_call_and_return_conditional_losses_1480s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџd@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџd@: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџd@
 
_user_specified_nameinputs:$ 

_user_specified_name2614:$ 

_user_specified_name2616
є	
ђ
A__inference_dense_2_layer_call_and_return_conditional_losses_2910

inputs0
matmul_readvariableop_resource:
 -
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
 *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ

 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
ў

џ
B__inference_conv3d_1_layer_call_and_return_conditional_losses_1395

inputs<
conv3d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:@@*
dtype0 
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:џџџџџџџџџd@*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:џџџџџџџџџd@m
IdentityIdentityBiasAdd:output:0^NoOp*
T0*5
_output_shapes#
!:џџџџџџџџџd@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:џџџџџџџџџd@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:] Y
5
_output_shapes#
!:џџџџџџџџџd@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Ћ

%__inference_conv3d_layer_call_fn_2234

inputs%
unknown:@
	unknown_0:@
identityЂStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:џџџџџџџџџd@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *I
fDRB
@__inference_conv3d_layer_call_and_return_conditional_losses_1330}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:џџџџџџџџџd@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:џџџџџџџџџd: : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:џџџџџџџџџd
 
_user_specified_nameinputs:$ 

_user_specified_name2228:$ 

_user_specified_name2230
ћ

%__inference_conv1d_layer_call_fn_2210

inputs
unknown:t@
	unknown_0:@
identityЂStatefulPartitionedCallо
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџd@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *I
fDRB
@__inference_conv1d_layer_call_and_return_conditional_losses_1350s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџd@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџdt: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџdt
 
_user_specified_nameinputs:$ 

_user_specified_name2204:$ 

_user_specified_name2206
Г&
Ѓ
)__inference_Full_Model_layer_call_fn_1850
input_1
input_2%
unknown:@
	unknown_0:@
	unknown_1:t@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@(

unknown_11:@@

unknown_12:@ 

unknown_13:@@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@(

unknown_23:@@

unknown_24:@ 

unknown_25:@@

unknown_26:@

unknown_27:@

unknown_28:@

unknown_29:@

unknown_30:@

unknown_31:@

unknown_32:@

unknown_33:@

unknown_34:@

unknown_35:@

unknown_36:

unknown_37:@

unknown_38:

unknown_39:
 

unknown_40: 

unknown_41: 

unknown_42:

unknown_43:

unknown_44:

unknown_45:

unknown_46:
identityЂStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46*=
Tin6
422*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*F
_read_only_resource_inputs(
&$$%&'()*+,-./01*2
config_proto" 

CPU

GPU2*0,1J 8 *M
fHRF
D__inference_Full_Model_layer_call_and_return_conditional_losses_1620o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*­
_input_shapes
:џџџџџџџџџdt:џџџџџџџџџd: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:џџџџџџџџџdt
!
_user_specified_name	input_1:^Z
5
_output_shapes#
!:џџџџџџџџџd
!
_user_specified_name	input_2:$ 

_user_specified_name1752:$ 

_user_specified_name1754:$ 

_user_specified_name1756:$ 

_user_specified_name1758:$ 

_user_specified_name1760:$ 

_user_specified_name1762:$ 

_user_specified_name1764:$	 

_user_specified_name1766:$
 

_user_specified_name1768:$ 

_user_specified_name1770:$ 

_user_specified_name1772:$ 

_user_specified_name1774:$ 

_user_specified_name1776:$ 

_user_specified_name1778:$ 

_user_specified_name1780:$ 

_user_specified_name1782:$ 

_user_specified_name1784:$ 

_user_specified_name1786:$ 

_user_specified_name1788:$ 

_user_specified_name1790:$ 

_user_specified_name1792:$ 

_user_specified_name1794:$ 

_user_specified_name1796:$ 

_user_specified_name1798:$ 

_user_specified_name1800:$ 

_user_specified_name1802:$ 

_user_specified_name1804:$ 

_user_specified_name1806:$ 

_user_specified_name1808:$ 

_user_specified_name1810:$  

_user_specified_name1812:$! 

_user_specified_name1814:$" 

_user_specified_name1816:$# 

_user_specified_name1818:$$ 

_user_specified_name1820:$% 

_user_specified_name1822:$& 

_user_specified_name1824:$' 

_user_specified_name1826:$( 

_user_specified_name1828:$) 

_user_specified_name1830:$* 

_user_specified_name1832:$+ 

_user_specified_name1834:$, 

_user_specified_name1836:$- 

_user_specified_name1838:$. 

_user_specified_name1840:$/ 

_user_specified_name1842:$0 

_user_specified_name1844:$1 

_user_specified_name1846
д
B
&__inference_re_lu_3_layer_call_fn_2401

inputs
identityП
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:џџџџџџџџџd@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *J
fERC
A__inference_re_lu_3_layer_call_and_return_conditional_losses_1378n
IdentityIdentityPartitionedCall:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџd@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:џџџџџџџџџd@:] Y
5
_output_shapes#
!:џџџџџџџџџd@
 
_user_specified_nameinputs
§
]
A__inference_re_lu_3_layer_call_and_return_conditional_losses_1378

inputs
identityT
ReluReluinputs*
T0*5
_output_shapes#
!:џџџџџџџџџd@h
IdentityIdentityRelu:activations:0*
T0*5
_output_shapes#
!:џџџџџџџџџd@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:џџџџџџџџџd@:] Y
5
_output_shapes#
!:џџџџџџџџџd@
 
_user_specified_nameinputs
д
B
&__inference_re_lu_4_layer_call_fn_2606

inputs
identityП
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:џџџџџџџџџd@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *J
fERC
A__inference_re_lu_4_layer_call_and_return_conditional_losses_1443n
IdentityIdentityPartitionedCall:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџd@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:џџџџџџџџџd@:] Y
5
_output_shapes#
!:џџџџџџџџџd@
 
_user_specified_nameinputs
щ

&__inference_dense_5_layer_call_fn_2957

inputs
unknown:
	unknown_0:
identityЂStatefulPartitionedCallл
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *J
fERC
A__inference_dense_5_layer_call_and_return_conditional_losses_1613o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:$ 

_user_specified_name2951:$ 

_user_specified_name2953
џ&
ш
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2509

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityЂAssignMovingAvgЂAssignMovingAvg/ReadVariableOpЂAssignMovingAvg_1Ђ AssignMovingAvg_1/ReadVariableOpЂbatchnorm/ReadVariableOpЂbatchnorm/mul/ReadVariableOpo
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(h
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:@
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       Ђ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@Ќ
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@Д
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@Ц
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:џџџџџџџџџџџџџџџџџџ@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
б
n
R__inference_global_average_pooling3d_layer_call_and_return_conditional_losses_2838

inputs
identityk
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ: {
W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
§
]
A__inference_re_lu_4_layer_call_and_return_conditional_losses_2611

inputs
identityT
ReluReluinputs*
T0*5
_output_shapes#
!:џџџџџџџџџd@h
IdentityIdentityRelu:activations:0*
T0*5
_output_shapes#
!:џџџџџџџџџd@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:џџџџџџџџџd@:] Y
5
_output_shapes#
!:џџџџџџџџџd@
 
_user_specified_nameinputs
џ&
ш
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2714

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityЂAssignMovingAvgЂAssignMovingAvg/ReadVariableOpЂAssignMovingAvg_1Ђ AssignMovingAvg_1/ReadVariableOpЂbatchnorm/ReadVariableOpЂbatchnorm/mul/ReadVariableOpo
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(h
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:@
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       Ђ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@Ќ
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@Д
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@Ц
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:џџџџџџџџџџџџџџџџџџ@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Ќ
B
&__inference_re_lu_2_layer_call_fn_2801

inputs
identityЕ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџd@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *J
fERC
A__inference_re_lu_2_layer_call_and_return_conditional_losses_1514d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:џџџџџџџџџd@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџd@:S O
+
_output_shapes
:џџџџџџџџџd@
 
_user_specified_nameinputs
Э

ђ
A__inference_dense_1_layer_call_and_return_conditional_losses_2878

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџS
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
е
О
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1109

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0ћ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*j
_output_shapesX
V:8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@:@:@:@:@:*
data_formatNDHWC*
epsilon%o:*
exponential_avg_factor%
з#<Ц
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(а
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*N
_output_shapes<
::8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@А
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:v r
N
_output_shapes<
::8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
З

Я
4__inference_batch_normalization_3_layer_call_fn_2337

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityЂStatefulPartitionedCallЇ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *W
fRRP
N__inference_batch_normalization_3_layer_call_and_return_conditional_losses_967
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*N
_output_shapes<
::8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs:$ 

_user_specified_name2327:$ 

_user_specified_name2329:$ 

_user_specified_name2331:$ 

_user_specified_name2333
х

$__inference_dense_layer_call_fn_2847

inputs
unknown:@
	unknown_0:
identityЂStatefulPartitionedCallй
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_1528o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs:$ 

_user_specified_name2841:$ 

_user_specified_name2843
К

Я
4__inference_batch_normalization_4_layer_call_fn_2555

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityЂStatefulPartitionedCallЊ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1127
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*N
_output_shapes<
::8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs:$ 

_user_specified_name2545:$ 

_user_specified_name2547:$ 

_user_specified_name2549:$ 

_user_specified_name2551
е
]
A__inference_re_lu_1_layer_call_and_return_conditional_losses_1449

inputs
identityJ
ReluReluinputs*
T0*+
_output_shapes
:џџџџџџџџџd@^
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:џџџџџџџџџd@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџd@:S O
+
_output_shapes
:џџџџџџџџџd@
 
_user_specified_nameinputs
б	
Я
4__inference_batch_normalization_1_layer_call_fn_2475

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1065|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:џџџџџџџџџџџџџџџџџџ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs:$ 

_user_specified_name2465:$ 

_user_specified_name2467:$ 

_user_specified_name2469:$ 

_user_specified_name2471
К

D__inference_Full_Model_layer_call_and_return_conditional_losses_1748
input_1
input_2)
conv3d_1624:@
conv3d_1626:@!
conv1d_1629:t@
conv1d_1631:@(
batch_normalization_3_1634:@(
batch_normalization_3_1636:@(
batch_normalization_3_1638:@(
batch_normalization_3_1640:@&
batch_normalization_1643:@&
batch_normalization_1645:@&
batch_normalization_1647:@&
batch_normalization_1649:@+
conv3d_1_1654:@@
conv3d_1_1656:@#
conv1d_1_1659:@@
conv1d_1_1661:@(
batch_normalization_4_1664:@(
batch_normalization_4_1666:@(
batch_normalization_4_1668:@(
batch_normalization_4_1670:@(
batch_normalization_1_1673:@(
batch_normalization_1_1675:@(
batch_normalization_1_1677:@(
batch_normalization_1_1679:@+
conv3d_2_1684:@@
conv3d_2_1686:@#
conv1d_2_1689:@@
conv1d_2_1691:@(
batch_normalization_5_1694:@(
batch_normalization_5_1696:@(
batch_normalization_5_1698:@(
batch_normalization_5_1700:@(
batch_normalization_2_1703:@(
batch_normalization_2_1705:@(
batch_normalization_2_1707:@(
batch_normalization_2_1709:@

dense_1716:@

dense_1718:
dense_1_1721:@
dense_1_1723:
dense_2_1727:
 
dense_2_1729: 
dense_3_1732: 
dense_3_1734:
dense_4_1737:
dense_4_1739:
dense_5_1742:
dense_5_1744:
identityЂ+batch_normalization/StatefulPartitionedCallЂ-batch_normalization_1/StatefulPartitionedCallЂ-batch_normalization_2/StatefulPartitionedCallЂ-batch_normalization_3/StatefulPartitionedCallЂ-batch_normalization_4/StatefulPartitionedCallЂ-batch_normalization_5/StatefulPartitionedCallЂconv1d/StatefulPartitionedCallЂ conv1d_1/StatefulPartitionedCallЂ conv1d_2/StatefulPartitionedCallЂconv3d/StatefulPartitionedCallЂ conv3d_1/StatefulPartitionedCallЂ conv3d_2/StatefulPartitionedCallЂdense/StatefulPartitionedCallЂdense_1/StatefulPartitionedCallЂdense_2/StatefulPartitionedCallЂdense_3/StatefulPartitionedCallЂdense_4/StatefulPartitionedCallЂdense_5/StatefulPartitionedCallі
conv3d/StatefulPartitionedCallStatefulPartitionedCallinput_2conv3d_1624conv3d_1626*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:џџџџџџџџџd@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *I
fDRB
@__inference_conv3d_layer_call_and_return_conditional_losses_1330ь
conv1d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv1d_1629conv1d_1631*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџd@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *I
fDRB
@__inference_conv1d_layer_call_and_return_conditional_losses_1350
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall'conv3d/StatefulPartitionedCall:output:0batch_normalization_3_1634batch_normalization_3_1636batch_normalization_3_1638batch_normalization_3_1640*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:џџџџџџџџџd@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *W
fRRP
N__inference_batch_normalization_3_layer_call_and_return_conditional_losses_985ї
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv1d/StatefulPartitionedCall:output:0batch_normalization_1643batch_normalization_1645batch_normalization_1647batch_normalization_1649*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџd@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *U
fPRN
L__inference_batch_normalization_layer_call_and_return_conditional_losses_923ї
re_lu_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:џџџџџџџџџd@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *J
fERC
A__inference_re_lu_3_layer_call_and_return_conditional_losses_1378ч
re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџd@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *H
fCRA
?__inference_re_lu_layer_call_and_return_conditional_losses_1384
 conv3d_1/StatefulPartitionedCallStatefulPartitionedCall re_lu_3/PartitionedCall:output:0conv3d_1_1654conv3d_1_1656*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:џџџџџџџџџd@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *K
fFRD
B__inference_conv3d_1_layer_call_and_return_conditional_losses_1395
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCallre_lu/PartitionedCall:output:0conv1d_1_1659conv1d_1_1661*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџd@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *K
fFRD
B__inference_conv1d_1_layer_call_and_return_conditional_losses_1415
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv3d_1/StatefulPartitionedCall:output:0batch_normalization_4_1664batch_normalization_4_1666batch_normalization_4_1668batch_normalization_4_1670*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:џџџџџџџџџd@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1127
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0batch_normalization_1_1673batch_normalization_1_1675batch_normalization_1_1677batch_normalization_1_1679*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџd@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1065ї
re_lu_4/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:џџџџџџџџџd@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *J
fERC
A__inference_re_lu_4_layer_call_and_return_conditional_losses_1443э
re_lu_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџd@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *J
fERC
A__inference_re_lu_1_layer_call_and_return_conditional_losses_1449
 conv3d_2/StatefulPartitionedCallStatefulPartitionedCall re_lu_4/PartitionedCall:output:0conv3d_2_1684conv3d_2_1686*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:џџџџџџџџџd@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *K
fFRD
B__inference_conv3d_2_layer_call_and_return_conditional_losses_1460
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall re_lu_1/PartitionedCall:output:0conv1d_2_1689conv1d_2_1691*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџd@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *K
fFRD
B__inference_conv1d_2_layer_call_and_return_conditional_losses_1480
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)conv3d_2/StatefulPartitionedCall:output:0batch_normalization_5_1694batch_normalization_5_1696batch_normalization_5_1698batch_normalization_5_1700*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:џџџџџџџџџd@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *X
fSRQ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1269
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0batch_normalization_2_1703batch_normalization_2_1705batch_normalization_2_1707batch_normalization_2_1709*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџd@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1207ї
re_lu_5/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:џџџџџџџџџd@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *J
fERC
A__inference_re_lu_5_layer_call_and_return_conditional_losses_1508э
re_lu_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџd@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *J
fERC
A__inference_re_lu_2_layer_call_and_return_conditional_losses_1514ѕ
(global_average_pooling3d/PartitionedCallPartitionedCall re_lu_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *[
fVRT
R__inference_global_average_pooling3d_layer_call_and_return_conditional_losses_1312ѕ
(global_average_pooling1d/PartitionedCallPartitionedCall re_lu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *[
fVRT
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_1301
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0
dense_1716
dense_1718*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_1528
dense_1/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling3d/PartitionedCall:output:0dense_1_1721dense_1_1723*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_1544
concatenate/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *N
fIRG
E__inference_concatenate_layer_call_and_return_conditional_losses_1556
dense_2/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_2_1727dense_2_1729*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_1567
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_1732dense_3_1734*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *J
fERC
A__inference_dense_3_layer_call_and_return_conditional_losses_1582
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_1737dense_4_1739*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *J
fERC
A__inference_dense_4_layer_call_and_return_conditional_losses_1597
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_1742dense_5_1744*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *J
fERC
A__inference_dense_5_layer_call_and_return_conditional_losses_1613w
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџи
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall^conv3d/StatefulPartitionedCall!^conv3d_1/StatefulPartitionedCall!^conv3d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*­
_input_shapes
:џџџџџџџџџdt:џџџџџџџџџd: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2@
conv3d/StatefulPartitionedCallconv3d/StatefulPartitionedCall2D
 conv3d_1/StatefulPartitionedCall conv3d_1/StatefulPartitionedCall2D
 conv3d_2/StatefulPartitionedCall conv3d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:T P
+
_output_shapes
:џџџџџџџџџdt
!
_user_specified_name	input_1:^Z
5
_output_shapes#
!:џџџџџџџџџd
!
_user_specified_name	input_2:$ 

_user_specified_name1624:$ 

_user_specified_name1626:$ 

_user_specified_name1629:$ 

_user_specified_name1631:$ 

_user_specified_name1634:$ 

_user_specified_name1636:$ 

_user_specified_name1638:$	 

_user_specified_name1640:$
 

_user_specified_name1643:$ 

_user_specified_name1645:$ 

_user_specified_name1647:$ 

_user_specified_name1649:$ 

_user_specified_name1654:$ 

_user_specified_name1656:$ 

_user_specified_name1659:$ 

_user_specified_name1661:$ 

_user_specified_name1664:$ 

_user_specified_name1666:$ 

_user_specified_name1668:$ 

_user_specified_name1670:$ 

_user_specified_name1673:$ 

_user_specified_name1675:$ 

_user_specified_name1677:$ 

_user_specified_name1679:$ 

_user_specified_name1684:$ 

_user_specified_name1686:$ 

_user_specified_name1689:$ 

_user_specified_name1691:$ 

_user_specified_name1694:$ 

_user_specified_name1696:$  

_user_specified_name1698:$! 

_user_specified_name1700:$" 

_user_specified_name1703:$# 

_user_specified_name1705:$$ 

_user_specified_name1707:$% 

_user_specified_name1709:$& 

_user_specified_name1716:$' 

_user_specified_name1718:$( 

_user_specified_name1721:$) 

_user_specified_name1723:$* 

_user_specified_name1727:$+ 

_user_specified_name1729:$, 

_user_specified_name1732:$- 

_user_specified_name1734:$. 

_user_specified_name1737:$/ 

_user_specified_name1739:$0 

_user_specified_name1742:$1 

_user_specified_name1744
Џ
 
'__inference_conv3d_1_layer_call_fn_2439

inputs%
unknown:@@
	unknown_0:@
identityЂStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:џџџџџџџџџd@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *K
fFRD
B__inference_conv3d_1_layer_call_and_return_conditional_losses_1395}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:џџџџџџџџџd@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:џџџџџџџџџd@: : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:џџџџџџџџџd@
 
_user_specified_nameinputs:$ 

_user_specified_name2433:$ 

_user_specified_name2435
є	
ђ
A__inference_dense_3_layer_call_and_return_conditional_losses_2929

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџS
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Й

Я
4__inference_batch_normalization_3_layer_call_fn_2350

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityЂStatefulPartitionedCallЉ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *W
fRRP
N__inference_batch_normalization_3_layer_call_and_return_conditional_losses_985
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*N
_output_shapes<
::8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs:$ 

_user_specified_name2340:$ 

_user_specified_name2342:$ 

_user_specified_name2344:$ 

_user_specified_name2346
щ

&__inference_dense_3_layer_call_fn_2919

inputs
unknown: 
	unknown_0:
identityЂStatefulPartitionedCallл
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *J
fERC
A__inference_dense_3_layer_call_and_return_conditional_losses_1582o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs:$ 

_user_specified_name2913:$ 

_user_specified_name2915
д
B
&__inference_re_lu_5_layer_call_fn_2811

inputs
identityП
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:џџџџџџџџџd@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *J
fERC
A__inference_re_lu_5_layer_call_and_return_conditional_losses_1508n
IdentityIdentityPartitionedCall:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџd@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:џџџџџџџџџd@:] Y
5
_output_shapes#
!:џџџџџџџџџd@
 
_user_specified_nameinputs
П&
Ѓ
)__inference_Full_Model_layer_call_fn_1952
input_1
input_2%
unknown:@
	unknown_0:@
	unknown_1:t@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@(

unknown_11:@@

unknown_12:@ 

unknown_13:@@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@(

unknown_23:@@

unknown_24:@ 

unknown_25:@@

unknown_26:@

unknown_27:@

unknown_28:@

unknown_29:@

unknown_30:@

unknown_31:@

unknown_32:@

unknown_33:@

unknown_34:@

unknown_35:@

unknown_36:

unknown_37:@

unknown_38:

unknown_39:
 

unknown_40: 

unknown_41: 

unknown_42:

unknown_43:

unknown_44:

unknown_45:

unknown_46:
identityЂStatefulPartitionedCallф
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46*=
Tin6
422*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*R
_read_only_resource_inputs4
20	
 !"#$%&'()*+,-./01*2
config_proto" 

CPU

GPU2*0,1J 8 *M
fHRF
D__inference_Full_Model_layer_call_and_return_conditional_losses_1748o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*­
_input_shapes
:џџџџџџџџџdt:џџџџџџџџџd: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:џџџџџџџџџdt
!
_user_specified_name	input_1:^Z
5
_output_shapes#
!:џџџџџџџџџd
!
_user_specified_name	input_2:$ 

_user_specified_name1854:$ 

_user_specified_name1856:$ 

_user_specified_name1858:$ 

_user_specified_name1860:$ 

_user_specified_name1862:$ 

_user_specified_name1864:$ 

_user_specified_name1866:$	 

_user_specified_name1868:$
 

_user_specified_name1870:$ 

_user_specified_name1872:$ 

_user_specified_name1874:$ 

_user_specified_name1876:$ 

_user_specified_name1878:$ 

_user_specified_name1880:$ 

_user_specified_name1882:$ 

_user_specified_name1884:$ 

_user_specified_name1886:$ 

_user_specified_name1888:$ 

_user_specified_name1890:$ 

_user_specified_name1892:$ 

_user_specified_name1894:$ 

_user_specified_name1896:$ 

_user_specified_name1898:$ 

_user_specified_name1900:$ 

_user_specified_name1902:$ 

_user_specified_name1904:$ 

_user_specified_name1906:$ 

_user_specified_name1908:$ 

_user_specified_name1910:$ 

_user_specified_name1912:$  

_user_specified_name1914:$! 

_user_specified_name1916:$" 

_user_specified_name1918:$# 

_user_specified_name1920:$$ 

_user_specified_name1922:$% 

_user_specified_name1924:$& 

_user_specified_name1926:$' 

_user_specified_name1928:$( 

_user_specified_name1930:$) 

_user_specified_name1932:$* 

_user_specified_name1934:$+ 

_user_specified_name1936:$, 

_user_specified_name1938:$- 

_user_specified_name1940:$. 

_user_specified_name1942:$/ 

_user_specified_name1944:$0 

_user_specified_name1946:$1 

_user_specified_name1948
щ

&__inference_dense_4_layer_call_fn_2938

inputs
unknown:
	unknown_0:
identityЂStatefulPartitionedCallл
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *J
fERC
A__inference_dense_4_layer_call_and_return_conditional_losses_1597o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:$ 

_user_specified_name2932:$ 

_user_specified_name2934
є	
ђ
A__inference_dense_4_layer_call_and_return_conditional_losses_2948

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџS
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
ў

џ
B__inference_conv3d_2_layer_call_and_return_conditional_losses_1460

inputs<
conv3d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:@@*
dtype0 
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:џџџџџџџџџd@*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:џџџџџџџџџd@m
IdentityIdentityBiasAdd:output:0^NoOp*
T0*5
_output_shapes#
!:џџџџџџџџџd@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:џџџџџџџџџd@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:] Y
5
_output_shapes#
!:џџџџџџџџџd@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
&

"__inference_signature_wrapper_2201
input_1
input_2%
unknown:@
	unknown_0:@
	unknown_1:t@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@(

unknown_11:@@

unknown_12:@ 

unknown_13:@@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@(

unknown_23:@@

unknown_24:@ 

unknown_25:@@

unknown_26:@

unknown_27:@

unknown_28:@

unknown_29:@

unknown_30:@

unknown_31:@

unknown_32:@

unknown_33:@

unknown_34:@

unknown_35:@

unknown_36:

unknown_37:@

unknown_38:

unknown_39:
 

unknown_40: 

unknown_41: 

unknown_42:

unknown_43:

unknown_44:

unknown_45:

unknown_46:
identityЂStatefulPartitionedCallО
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46*=
Tin6
422*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*R
_read_only_resource_inputs4
20	
 !"#$%&'()*+,-./01*2
config_proto" 

CPU

GPU2*0,1J 8 *'
f"R 
__inference__wrapped_model_869o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*­
_input_shapes
:џџџџџџџџџdt:џџџџџџџџџd: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:џџџџџџџџџdt
!
_user_specified_name	input_1:^Z
5
_output_shapes#
!:џџџџџџџџџd
!
_user_specified_name	input_2:$ 

_user_specified_name2103:$ 

_user_specified_name2105:$ 

_user_specified_name2107:$ 

_user_specified_name2109:$ 

_user_specified_name2111:$ 

_user_specified_name2113:$ 

_user_specified_name2115:$	 

_user_specified_name2117:$
 

_user_specified_name2119:$ 

_user_specified_name2121:$ 

_user_specified_name2123:$ 

_user_specified_name2125:$ 

_user_specified_name2127:$ 

_user_specified_name2129:$ 

_user_specified_name2131:$ 

_user_specified_name2133:$ 

_user_specified_name2135:$ 

_user_specified_name2137:$ 

_user_specified_name2139:$ 

_user_specified_name2141:$ 

_user_specified_name2143:$ 

_user_specified_name2145:$ 

_user_specified_name2147:$ 

_user_specified_name2149:$ 

_user_specified_name2151:$ 

_user_specified_name2153:$ 

_user_specified_name2155:$ 

_user_specified_name2157:$ 

_user_specified_name2159:$ 

_user_specified_name2161:$  

_user_specified_name2163:$! 

_user_specified_name2165:$" 

_user_specified_name2167:$# 

_user_specified_name2169:$$ 

_user_specified_name2171:$% 

_user_specified_name2173:$& 

_user_specified_name2175:$' 

_user_specified_name2177:$( 

_user_specified_name2179:$) 

_user_specified_name2181:$* 

_user_specified_name2183:$+ 

_user_specified_name2185:$, 

_user_specified_name2187:$- 

_user_specified_name2189:$. 

_user_specified_name2191:$/ 

_user_specified_name2193:$0 

_user_specified_name2195:$1 

_user_specified_name2197
е
О
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_2368

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0ћ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*j
_output_shapesX
V:8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@:@:@:@:@:*
data_formatNDHWC*
epsilon%o:*
exponential_avg_factor%
з#<Ц
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(а
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*N
_output_shapes<
::8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@А
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:v r
N
_output_shapes<
::8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
е
]
A__inference_re_lu_2_layer_call_and_return_conditional_losses_2806

inputs
identityJ
ReluReluinputs*
T0*+
_output_shapes
:џџџџџџџџџd@^
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:џџџџџџџџџd@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџd@:S O
+
_output_shapes
:џџџџџџџџџd@
 
_user_specified_nameinputs
е
О
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1251

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0ћ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*j
_output_shapesX
V:8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@:@:@:@:@:*
data_formatNDHWC*
epsilon%o:*
exponential_avg_factor%
з#<Ц
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(а
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*N
_output_shapes<
::8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@А
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:v r
N
_output_shapes<
::8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
К

Я
4__inference_batch_normalization_5_layer_call_fn_2760

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityЂStatefulPartitionedCallЊ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *X
fSRQ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1269
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*N
_output_shapes<
::8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs:$ 

_user_specified_name2750:$ 

_user_specified_name2752:$ 

_user_specified_name2754:$ 

_user_specified_name2756
г
[
?__inference_re_lu_layer_call_and_return_conditional_losses_1384

inputs
identityJ
ReluReluinputs*
T0*+
_output_shapes
:џџџџџџџџџd@^
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:џџџџџџџџџd@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџd@:S O
+
_output_shapes
:џџџџџџџџџd@
 
_user_specified_nameinputs

n
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_1301

inputs
identityX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ў

D__inference_Full_Model_layer_call_and_return_conditional_losses_1620
input_1
input_2)
conv3d_1331:@
conv3d_1333:@!
conv1d_1351:t@
conv1d_1353:@(
batch_normalization_3_1356:@(
batch_normalization_3_1358:@(
batch_normalization_3_1360:@(
batch_normalization_3_1362:@&
batch_normalization_1365:@&
batch_normalization_1367:@&
batch_normalization_1369:@&
batch_normalization_1371:@+
conv3d_1_1396:@@
conv3d_1_1398:@#
conv1d_1_1416:@@
conv1d_1_1418:@(
batch_normalization_4_1421:@(
batch_normalization_4_1423:@(
batch_normalization_4_1425:@(
batch_normalization_4_1427:@(
batch_normalization_1_1430:@(
batch_normalization_1_1432:@(
batch_normalization_1_1434:@(
batch_normalization_1_1436:@+
conv3d_2_1461:@@
conv3d_2_1463:@#
conv1d_2_1481:@@
conv1d_2_1483:@(
batch_normalization_5_1486:@(
batch_normalization_5_1488:@(
batch_normalization_5_1490:@(
batch_normalization_5_1492:@(
batch_normalization_2_1495:@(
batch_normalization_2_1497:@(
batch_normalization_2_1499:@(
batch_normalization_2_1501:@

dense_1529:@

dense_1531:
dense_1_1545:@
dense_1_1547:
dense_2_1568:
 
dense_2_1570: 
dense_3_1583: 
dense_3_1585:
dense_4_1598:
dense_4_1600:
dense_5_1614:
dense_5_1616:
identityЂ+batch_normalization/StatefulPartitionedCallЂ-batch_normalization_1/StatefulPartitionedCallЂ-batch_normalization_2/StatefulPartitionedCallЂ-batch_normalization_3/StatefulPartitionedCallЂ-batch_normalization_4/StatefulPartitionedCallЂ-batch_normalization_5/StatefulPartitionedCallЂconv1d/StatefulPartitionedCallЂ conv1d_1/StatefulPartitionedCallЂ conv1d_2/StatefulPartitionedCallЂconv3d/StatefulPartitionedCallЂ conv3d_1/StatefulPartitionedCallЂ conv3d_2/StatefulPartitionedCallЂdense/StatefulPartitionedCallЂdense_1/StatefulPartitionedCallЂdense_2/StatefulPartitionedCallЂdense_3/StatefulPartitionedCallЂdense_4/StatefulPartitionedCallЂdense_5/StatefulPartitionedCallі
conv3d/StatefulPartitionedCallStatefulPartitionedCallinput_2conv3d_1331conv3d_1333*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:џџџџџџџџџd@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *I
fDRB
@__inference_conv3d_layer_call_and_return_conditional_losses_1330ь
conv1d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv1d_1351conv1d_1353*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџd@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *I
fDRB
@__inference_conv1d_layer_call_and_return_conditional_losses_1350
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall'conv3d/StatefulPartitionedCall:output:0batch_normalization_3_1356batch_normalization_3_1358batch_normalization_3_1360batch_normalization_3_1362*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:џџџџџџџџџd@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *W
fRRP
N__inference_batch_normalization_3_layer_call_and_return_conditional_losses_967ѕ
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv1d/StatefulPartitionedCall:output:0batch_normalization_1365batch_normalization_1367batch_normalization_1369batch_normalization_1371*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџd@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *U
fPRN
L__inference_batch_normalization_layer_call_and_return_conditional_losses_903ї
re_lu_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:џџџџџџџџџd@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *J
fERC
A__inference_re_lu_3_layer_call_and_return_conditional_losses_1378ч
re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџd@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *H
fCRA
?__inference_re_lu_layer_call_and_return_conditional_losses_1384
 conv3d_1/StatefulPartitionedCallStatefulPartitionedCall re_lu_3/PartitionedCall:output:0conv3d_1_1396conv3d_1_1398*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:џџџџџџџџџd@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *K
fFRD
B__inference_conv3d_1_layer_call_and_return_conditional_losses_1395
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCallre_lu/PartitionedCall:output:0conv1d_1_1416conv1d_1_1418*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџd@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *K
fFRD
B__inference_conv1d_1_layer_call_and_return_conditional_losses_1415
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv3d_1/StatefulPartitionedCall:output:0batch_normalization_4_1421batch_normalization_4_1423batch_normalization_4_1425batch_normalization_4_1427*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:џџџџџџџџџd@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1109
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0batch_normalization_1_1430batch_normalization_1_1432batch_normalization_1_1434batch_normalization_1_1436*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџd@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1045ї
re_lu_4/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:џџџџџџџџџd@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *J
fERC
A__inference_re_lu_4_layer_call_and_return_conditional_losses_1443э
re_lu_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџd@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *J
fERC
A__inference_re_lu_1_layer_call_and_return_conditional_losses_1449
 conv3d_2/StatefulPartitionedCallStatefulPartitionedCall re_lu_4/PartitionedCall:output:0conv3d_2_1461conv3d_2_1463*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:џџџџџџџџџd@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *K
fFRD
B__inference_conv3d_2_layer_call_and_return_conditional_losses_1460
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall re_lu_1/PartitionedCall:output:0conv1d_2_1481conv1d_2_1483*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџd@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *K
fFRD
B__inference_conv1d_2_layer_call_and_return_conditional_losses_1480
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)conv3d_2/StatefulPartitionedCall:output:0batch_normalization_5_1486batch_normalization_5_1488batch_normalization_5_1490batch_normalization_5_1492*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:џџџџџџџџџd@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *X
fSRQ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1251
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0batch_normalization_2_1495batch_normalization_2_1497batch_normalization_2_1499batch_normalization_2_1501*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџd@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1187ї
re_lu_5/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:џџџџџџџџџd@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *J
fERC
A__inference_re_lu_5_layer_call_and_return_conditional_losses_1508э
re_lu_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџd@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *J
fERC
A__inference_re_lu_2_layer_call_and_return_conditional_losses_1514ѕ
(global_average_pooling3d/PartitionedCallPartitionedCall re_lu_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *[
fVRT
R__inference_global_average_pooling3d_layer_call_and_return_conditional_losses_1312ѕ
(global_average_pooling1d/PartitionedCallPartitionedCall re_lu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *[
fVRT
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_1301
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0
dense_1529
dense_1531*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_1528
dense_1/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling3d/PartitionedCall:output:0dense_1_1545dense_1_1547*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_1544
concatenate/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *N
fIRG
E__inference_concatenate_layer_call_and_return_conditional_losses_1556
dense_2/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_2_1568dense_2_1570*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_1567
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_1583dense_3_1585*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *J
fERC
A__inference_dense_3_layer_call_and_return_conditional_losses_1582
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_1598dense_4_1600*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *J
fERC
A__inference_dense_4_layer_call_and_return_conditional_losses_1597
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_1614dense_5_1616*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *J
fERC
A__inference_dense_5_layer_call_and_return_conditional_losses_1613w
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџи
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall^conv3d/StatefulPartitionedCall!^conv3d_1/StatefulPartitionedCall!^conv3d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*­
_input_shapes
:џџџџџџџџџdt:џџџџџџџџџd: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2@
conv3d/StatefulPartitionedCallconv3d/StatefulPartitionedCall2D
 conv3d_1/StatefulPartitionedCall conv3d_1/StatefulPartitionedCall2D
 conv3d_2/StatefulPartitionedCall conv3d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:T P
+
_output_shapes
:џџџџџџџџџdt
!
_user_specified_name	input_1:^Z
5
_output_shapes#
!:џџџџџџџџџd
!
_user_specified_name	input_2:$ 

_user_specified_name1331:$ 

_user_specified_name1333:$ 

_user_specified_name1351:$ 

_user_specified_name1353:$ 

_user_specified_name1356:$ 

_user_specified_name1358:$ 

_user_specified_name1360:$	 

_user_specified_name1362:$
 

_user_specified_name1365:$ 

_user_specified_name1367:$ 

_user_specified_name1369:$ 

_user_specified_name1371:$ 

_user_specified_name1396:$ 

_user_specified_name1398:$ 

_user_specified_name1416:$ 

_user_specified_name1418:$ 

_user_specified_name1421:$ 

_user_specified_name1423:$ 

_user_specified_name1425:$ 

_user_specified_name1427:$ 

_user_specified_name1430:$ 

_user_specified_name1432:$ 

_user_specified_name1434:$ 

_user_specified_name1436:$ 

_user_specified_name1461:$ 

_user_specified_name1463:$ 

_user_specified_name1481:$ 

_user_specified_name1483:$ 

_user_specified_name1486:$ 

_user_specified_name1488:$  

_user_specified_name1490:$! 

_user_specified_name1492:$" 

_user_specified_name1495:$# 

_user_specified_name1497:$$ 

_user_specified_name1499:$% 

_user_specified_name1501:$& 

_user_specified_name1529:$' 

_user_specified_name1531:$( 

_user_specified_name1545:$) 

_user_specified_name1547:$* 

_user_specified_name1568:$+ 

_user_specified_name1570:$, 

_user_specified_name1583:$- 

_user_specified_name1585:$. 

_user_specified_name1598:$/ 

_user_specified_name1600:$0 

_user_specified_name1614:$1 

_user_specified_name1616
А
S
7__inference_global_average_pooling3d_layer_call_fn_2832

inputs
identityЫ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *[
fVRT
R__inference_global_average_pooling3d_layer_call_and_return_conditional_losses_1312i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ: {
W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Э

ђ
A__inference_dense_5_layer_call_and_return_conditional_losses_1613

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџS
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Љ
V
*__inference_concatenate_layer_call_fn_2884
inputs_0
inputs_1
identityТ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *N
fIRG
E__inference_concatenate_layer_call_and_return_conditional_losses_1556`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџ:џџџџџџџџџ:Q M
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_0:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_1


O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1127

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0э
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*j
_output_shapesX
V:8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@:@:@:@:@:*
data_formatNDHWC*
epsilon%o:*
is_training( 
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*N
_output_shapes<
::8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:v r
N
_output_shapes<
::8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource

Ў
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2529

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identityЂbatchnorm/ReadVariableOpЂbatchnorm/ReadVariableOp_1Ђbatchnorm/ReadVariableOp_2Ђbatchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:џџџџџџџџџџџџџџџџџџ@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
ќ

§
@__inference_conv3d_layer_call_and_return_conditional_losses_2244

inputs<
conv3d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:@*
dtype0 
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:џџџџџџџџџd@*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:џџџџџџџџџd@m
IdentityIdentityBiasAdd:output:0^NoOp*
T0*5
_output_shapes#
!:џџџџџџџџџd@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:џџџџџџџџџd: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:] Y
5
_output_shapes#
!:џџџџџџџџџd
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource


O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_2796

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0э
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*j
_output_shapesX
V:8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@:@:@:@:@:*
data_formatNDHWC*
epsilon%o:*
is_training( 
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*N
_output_shapes<
::8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:v r
N
_output_shapes<
::8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
И

Я
4__inference_batch_normalization_4_layer_call_fn_2542

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityЂStatefulPartitionedCallЈ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1109
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*N
_output_shapes<
::8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs:$ 

_user_specified_name2532:$ 

_user_specified_name2534:$ 

_user_specified_name2536:$ 

_user_specified_name2538
Ј
@
$__inference_re_lu_layer_call_fn_2391

inputs
identityГ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџd@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *H
fCRA
?__inference_re_lu_layer_call_and_return_conditional_losses_1384d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:џџџџџџџџџd@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџd@:S O
+
_output_shapes
:џџџџџџџџџd@
 
_user_specified_nameinputs
е
О
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_2573

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0ћ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*j
_output_shapesX
V:8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@:@:@:@:@:*
data_formatNDHWC*
epsilon%o:*
exponential_avg_factor%
з#<Ц
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(а
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*N
_output_shapes<
::8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@А
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:v r
N
_output_shapes<
::8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
§
]
A__inference_re_lu_4_layer_call_and_return_conditional_losses_1443

inputs
identityT
ReluReluinputs*
T0*5
_output_shapes#
!:џџџџџџџџџd@h
IdentityIdentityRelu:activations:0*
T0*5
_output_shapes#
!:џџџџџџџџџd@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:џџџџџџџџџd@:] Y
5
_output_shapes#
!:џџџџџџџџџd@
 
_user_specified_nameinputs
є	
ђ
A__inference_dense_4_layer_call_and_return_conditional_losses_1597

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџS
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
ќ
S
7__inference_global_average_pooling1d_layer_call_fn_2821

inputs
identityЫ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *[
fVRT
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_1301i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs


B__inference_conv1d_1_layer_call_and_return_conditional_losses_1415

inputsA
+conv1d_expanddims_1_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂ"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџd@
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :  
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@Ќ
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџd@*
paddingSAME*
strides

Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:џџџџџџџџџd@*
squeeze_dims

§џџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџd@c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџd@`
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџd@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:џџџџџџџџџd@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource


O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1269

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0э
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*j
_output_shapesX
V:8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@:@:@:@:@:*
data_formatNDHWC*
epsilon%o:*
is_training( 
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*N
_output_shapes<
::8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:v r
N
_output_shapes<
::8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource


B__inference_conv1d_2_layer_call_and_return_conditional_losses_1480

inputsA
+conv1d_expanddims_1_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂ"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџd@
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :  
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@Ќ
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџd@*
paddingSAME*
strides

Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:џџџџџџџџџd@*
squeeze_dims

§џџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџd@c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџd@`
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџd@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:џџџџџџџџџd@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource

Ў
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1207

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identityЂbatchnorm/ReadVariableOpЂbatchnorm/ReadVariableOp_1Ђbatchnorm/ReadVariableOp_2Ђbatchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:џџџџџџџџџџџџџџџџџџ@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
џ

'__inference_conv1d_1_layer_call_fn_2415

inputs
unknown:@@
	unknown_0:@
identityЂStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџd@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *K
fFRD
B__inference_conv1d_1_layer_call_and_return_conditional_losses_1415s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџd@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџd@: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџd@
 
_user_specified_nameinputs:$ 

_user_specified_name2409:$ 

_user_specified_name2411


@__inference_conv1d_layer_call_and_return_conditional_losses_2225

inputsA
+conv1d_expanddims_1_readvariableop_resource:t@-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂ"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџdt
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:t@*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :  
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:t@Ќ
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџd@*
paddingSAME*
strides

Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:џџџџџџџџџd@*
squeeze_dims

§џџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџd@c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџd@`
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџdt: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:џџџџџџџџџdt
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Ж
o
E__inference_concatenate_layer_call_and_return_conditional_losses_1556

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :u
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ
W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџ:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

Ў
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1065

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identityЂbatchnorm/ReadVariableOpЂbatchnorm/ReadVariableOp_1Ђbatchnorm/ReadVariableOp_2Ђbatchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:џџџџџџџџџџџџџџџџџџ@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource

Ў
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2734

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identityЂbatchnorm/ReadVariableOpЂbatchnorm/ReadVariableOp_1Ђbatchnorm/ReadVariableOp_2Ђbatchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:џџџџџџџџџџџџџџџџџџ@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Ы

№
?__inference_dense_layer_call_and_return_conditional_losses_2858

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџS
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Ь	
Э
2__inference_batch_normalization_layer_call_fn_2270

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *U
fPRN
L__inference_batch_normalization_layer_call_and_return_conditional_losses_923|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:џџџџџџџџџџџџџџџџџџ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs:$ 

_user_specified_name2260:$ 

_user_specified_name2262:$ 

_user_specified_name2264:$ 

_user_specified_name2266
є	
ђ
A__inference_dense_3_layer_call_and_return_conditional_losses_1582

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџS
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
џ&
ш
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1045

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityЂAssignMovingAvgЂAssignMovingAvg/ReadVariableOpЂAssignMovingAvg_1Ђ AssignMovingAvg_1/ReadVariableOpЂbatchnorm/ReadVariableOpЂbatchnorm/mul/ReadVariableOpo
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(h
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:@
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       Ђ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@Ќ
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@Д
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@Ц
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:џџџџџџџџџџџџџџџџџџ@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource

Ќ
M__inference_batch_normalization_layer_call_and_return_conditional_losses_2324

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identityЂbatchnorm/ReadVariableOpЂbatchnorm/ReadVariableOp_1Ђbatchnorm/ReadVariableOp_2Ђbatchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:џџџџџџџџџџџџџџџџџџ@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
И

Я
4__inference_batch_normalization_5_layer_call_fn_2747

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityЂStatefulPartitionedCallЈ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *X
fSRQ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1251
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*N
_output_shapes<
::8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs:$ 

_user_specified_name2737:$ 

_user_specified_name2739:$ 

_user_specified_name2741:$ 

_user_specified_name2743
§&
ц
M__inference_batch_normalization_layer_call_and_return_conditional_losses_2304

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityЂAssignMovingAvgЂAssignMovingAvg/ReadVariableOpЂAssignMovingAvg_1Ђ AssignMovingAvg_1/ReadVariableOpЂbatchnorm/ReadVariableOpЂbatchnorm/mul/ReadVariableOpo
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(h
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:@
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       Ђ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@Ќ
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@Д
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@Ц
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:џџџџџџџџџџџџџџџџџџ@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Э

ђ
A__inference_dense_5_layer_call_and_return_conditional_losses_2968

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџS
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource


O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_2591

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0э
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*j
_output_shapesX
V:8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@:@:@:@:@:*
data_formatNDHWC*
epsilon%o:*
is_training( 
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*N
_output_shapes<
::8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:v r
N
_output_shapes<
::8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Ќ
B
&__inference_re_lu_1_layer_call_fn_2596

inputs
identityЕ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџd@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *J
fERC
A__inference_re_lu_1_layer_call_and_return_conditional_losses_1449d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:џџџџџџџџџd@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџd@:S O
+
_output_shapes
:џџџџџџџџџd@
 
_user_specified_nameinputs
ў

џ
B__inference_conv3d_2_layer_call_and_return_conditional_losses_2654

inputs<
conv3d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:@@*
dtype0 
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:џџџџџџџџџd@*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:џџџџџџџџџd@m
IdentityIdentityBiasAdd:output:0^NoOp*
T0*5
_output_shapes#
!:џџџџџџџџџd@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:џџџџџџџџџd@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:] Y
5
_output_shapes#
!:џџџџџџџџџd@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
§
]
A__inference_re_lu_3_layer_call_and_return_conditional_losses_2406

inputs
identityT
ReluReluinputs*
T0*5
_output_shapes#
!:џџџџџџџџџd@h
IdentityIdentityRelu:activations:0*
T0*5
_output_shapes#
!:џџџџџџџџџd@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:џџџџџџџџџd@:] Y
5
_output_shapes#
!:џџџџџџџџџd@
 
_user_specified_nameinputs
Э

ђ
A__inference_dense_1_layer_call_and_return_conditional_losses_1544

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџS
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource


N__inference_batch_normalization_3_layer_call_and_return_conditional_losses_985

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0э
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*j
_output_shapesX
V:8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@:@:@:@:@:*
data_formatNDHWC*
epsilon%o:*
is_training( 
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*N
_output_shapes<
::8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:v r
N
_output_shapes<
::8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource


@__inference_conv1d_layer_call_and_return_conditional_losses_1350

inputsA
+conv1d_expanddims_1_readvariableop_resource:t@-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂ"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџdt
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:t@*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :  
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:t@Ќ
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџd@*
paddingSAME*
strides

Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:џџџџџџџџџd@*
squeeze_dims

§џџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџd@c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџd@`
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџdt: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:џџџџџџџџџdt
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
џ&
ш
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1187

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityЂAssignMovingAvgЂAssignMovingAvg/ReadVariableOpЂAssignMovingAvg_1Ђ AssignMovingAvg_1/ReadVariableOpЂbatchnorm/ReadVariableOpЂbatchnorm/mul/ReadVariableOpo
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(h
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:@
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       Ђ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@Ќ
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@Д
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@Ц
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:џџџџџџџџџџџџџџџџџџ@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
г
[
?__inference_re_lu_layer_call_and_return_conditional_losses_2396

inputs
identityJ
ReluReluinputs*
T0*+
_output_shapes
:џџџџџџџџџd@^
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:џџџџџџџџџd@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџd@:S O
+
_output_shapes
:џџџџџџџџџd@
 
_user_specified_nameinputs
ЂМ
1
__inference__wrapped_model_869
input_1
input_2N
0full_model_conv3d_conv3d_readvariableop_resource:@?
1full_model_conv3d_biasadd_readvariableop_resource:@S
=full_model_conv1d_conv1d_expanddims_1_readvariableop_resource:t@?
1full_model_conv1d_biasadd_readvariableop_resource:@F
8full_model_batch_normalization_3_readvariableop_resource:@H
:full_model_batch_normalization_3_readvariableop_1_resource:@W
Ifull_model_batch_normalization_3_fusedbatchnormv3_readvariableop_resource:@Y
Kfull_model_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:@N
@full_model_batch_normalization_batchnorm_readvariableop_resource:@R
Dfull_model_batch_normalization_batchnorm_mul_readvariableop_resource:@P
Bfull_model_batch_normalization_batchnorm_readvariableop_1_resource:@P
Bfull_model_batch_normalization_batchnorm_readvariableop_2_resource:@P
2full_model_conv3d_1_conv3d_readvariableop_resource:@@A
3full_model_conv3d_1_biasadd_readvariableop_resource:@U
?full_model_conv1d_1_conv1d_expanddims_1_readvariableop_resource:@@A
3full_model_conv1d_1_biasadd_readvariableop_resource:@F
8full_model_batch_normalization_4_readvariableop_resource:@H
:full_model_batch_normalization_4_readvariableop_1_resource:@W
Ifull_model_batch_normalization_4_fusedbatchnormv3_readvariableop_resource:@Y
Kfull_model_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:@P
Bfull_model_batch_normalization_1_batchnorm_readvariableop_resource:@T
Ffull_model_batch_normalization_1_batchnorm_mul_readvariableop_resource:@R
Dfull_model_batch_normalization_1_batchnorm_readvariableop_1_resource:@R
Dfull_model_batch_normalization_1_batchnorm_readvariableop_2_resource:@P
2full_model_conv3d_2_conv3d_readvariableop_resource:@@A
3full_model_conv3d_2_biasadd_readvariableop_resource:@U
?full_model_conv1d_2_conv1d_expanddims_1_readvariableop_resource:@@A
3full_model_conv1d_2_biasadd_readvariableop_resource:@F
8full_model_batch_normalization_5_readvariableop_resource:@H
:full_model_batch_normalization_5_readvariableop_1_resource:@W
Ifull_model_batch_normalization_5_fusedbatchnormv3_readvariableop_resource:@Y
Kfull_model_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:@P
Bfull_model_batch_normalization_2_batchnorm_readvariableop_resource:@T
Ffull_model_batch_normalization_2_batchnorm_mul_readvariableop_resource:@R
Dfull_model_batch_normalization_2_batchnorm_readvariableop_1_resource:@R
Dfull_model_batch_normalization_2_batchnorm_readvariableop_2_resource:@A
/full_model_dense_matmul_readvariableop_resource:@>
0full_model_dense_biasadd_readvariableop_resource:C
1full_model_dense_1_matmul_readvariableop_resource:@@
2full_model_dense_1_biasadd_readvariableop_resource:C
1full_model_dense_2_matmul_readvariableop_resource:
 @
2full_model_dense_2_biasadd_readvariableop_resource: C
1full_model_dense_3_matmul_readvariableop_resource: @
2full_model_dense_3_biasadd_readvariableop_resource:C
1full_model_dense_4_matmul_readvariableop_resource:@
2full_model_dense_4_biasadd_readvariableop_resource:C
1full_model_dense_5_matmul_readvariableop_resource:@
2full_model_dense_5_biasadd_readvariableop_resource:
identityЂ7Full_Model/batch_normalization/batchnorm/ReadVariableOpЂ9Full_Model/batch_normalization/batchnorm/ReadVariableOp_1Ђ9Full_Model/batch_normalization/batchnorm/ReadVariableOp_2Ђ;Full_Model/batch_normalization/batchnorm/mul/ReadVariableOpЂ9Full_Model/batch_normalization_1/batchnorm/ReadVariableOpЂ;Full_Model/batch_normalization_1/batchnorm/ReadVariableOp_1Ђ;Full_Model/batch_normalization_1/batchnorm/ReadVariableOp_2Ђ=Full_Model/batch_normalization_1/batchnorm/mul/ReadVariableOpЂ9Full_Model/batch_normalization_2/batchnorm/ReadVariableOpЂ;Full_Model/batch_normalization_2/batchnorm/ReadVariableOp_1Ђ;Full_Model/batch_normalization_2/batchnorm/ReadVariableOp_2Ђ=Full_Model/batch_normalization_2/batchnorm/mul/ReadVariableOpЂ@Full_Model/batch_normalization_3/FusedBatchNormV3/ReadVariableOpЂBFull_Model/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Ђ/Full_Model/batch_normalization_3/ReadVariableOpЂ1Full_Model/batch_normalization_3/ReadVariableOp_1Ђ@Full_Model/batch_normalization_4/FusedBatchNormV3/ReadVariableOpЂBFull_Model/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Ђ/Full_Model/batch_normalization_4/ReadVariableOpЂ1Full_Model/batch_normalization_4/ReadVariableOp_1Ђ@Full_Model/batch_normalization_5/FusedBatchNormV3/ReadVariableOpЂBFull_Model/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1Ђ/Full_Model/batch_normalization_5/ReadVariableOpЂ1Full_Model/batch_normalization_5/ReadVariableOp_1Ђ(Full_Model/conv1d/BiasAdd/ReadVariableOpЂ4Full_Model/conv1d/Conv1D/ExpandDims_1/ReadVariableOpЂ*Full_Model/conv1d_1/BiasAdd/ReadVariableOpЂ6Full_Model/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOpЂ*Full_Model/conv1d_2/BiasAdd/ReadVariableOpЂ6Full_Model/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOpЂ(Full_Model/conv3d/BiasAdd/ReadVariableOpЂ'Full_Model/conv3d/Conv3D/ReadVariableOpЂ*Full_Model/conv3d_1/BiasAdd/ReadVariableOpЂ)Full_Model/conv3d_1/Conv3D/ReadVariableOpЂ*Full_Model/conv3d_2/BiasAdd/ReadVariableOpЂ)Full_Model/conv3d_2/Conv3D/ReadVariableOpЂ'Full_Model/dense/BiasAdd/ReadVariableOpЂ&Full_Model/dense/MatMul/ReadVariableOpЂ)Full_Model/dense_1/BiasAdd/ReadVariableOpЂ(Full_Model/dense_1/MatMul/ReadVariableOpЂ)Full_Model/dense_2/BiasAdd/ReadVariableOpЂ(Full_Model/dense_2/MatMul/ReadVariableOpЂ)Full_Model/dense_3/BiasAdd/ReadVariableOpЂ(Full_Model/dense_3/MatMul/ReadVariableOpЂ)Full_Model/dense_4/BiasAdd/ReadVariableOpЂ(Full_Model/dense_4/MatMul/ReadVariableOpЂ)Full_Model/dense_5/BiasAdd/ReadVariableOpЂ(Full_Model/dense_5/MatMul/ReadVariableOpЄ
'Full_Model/conv3d/Conv3D/ReadVariableOpReadVariableOp0full_model_conv3d_conv3d_readvariableop_resource**
_output_shapes
:@*
dtype0Х
Full_Model/conv3d/Conv3DConv3Dinput_2/Full_Model/conv3d/Conv3D/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:џџџџџџџџџd@*
paddingSAME*
strides	

(Full_Model/conv3d/BiasAdd/ReadVariableOpReadVariableOp1full_model_conv3d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Й
Full_Model/conv3d/BiasAddBiasAdd!Full_Model/conv3d/Conv3D:output:00Full_Model/conv3d/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:џџџџџџџџџd@r
'Full_Model/conv1d/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџІ
#Full_Model/conv1d/Conv1D/ExpandDims
ExpandDimsinput_10Full_Model/conv1d/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџdtЖ
4Full_Model/conv1d/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp=full_model_conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:t@*
dtype0k
)Full_Model/conv1d/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ж
%Full_Model/conv1d/Conv1D/ExpandDims_1
ExpandDims<Full_Model/conv1d/Conv1D/ExpandDims_1/ReadVariableOp:value:02Full_Model/conv1d/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:t@т
Full_Model/conv1d/Conv1DConv2D,Full_Model/conv1d/Conv1D/ExpandDims:output:0.Full_Model/conv1d/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџd@*
paddingSAME*
strides
Є
 Full_Model/conv1d/Conv1D/SqueezeSqueeze!Full_Model/conv1d/Conv1D:output:0*
T0*+
_output_shapes
:џџџџџџџџџd@*
squeeze_dims

§џџџџџџџџ
(Full_Model/conv1d/BiasAdd/ReadVariableOpReadVariableOp1full_model_conv1d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0З
Full_Model/conv1d/BiasAddBiasAdd)Full_Model/conv1d/Conv1D/Squeeze:output:00Full_Model/conv1d/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџd@Є
/Full_Model/batch_normalization_3/ReadVariableOpReadVariableOp8full_model_batch_normalization_3_readvariableop_resource*
_output_shapes
:@*
dtype0Ј
1Full_Model/batch_normalization_3/ReadVariableOp_1ReadVariableOp:full_model_batch_normalization_3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ц
@Full_Model/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpIfull_model_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0Ъ
BFull_Model/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpKfull_model_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0
1Full_Model/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3"Full_Model/conv3d/BiasAdd:output:07Full_Model/batch_normalization_3/ReadVariableOp:value:09Full_Model/batch_normalization_3/ReadVariableOp_1:value:0HFull_Model/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0JFull_Model/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*Q
_output_shapes?
=:џџџџџџџџџd@:@:@:@:@:*
data_formatNDHWC*
epsilon%o:*
is_training( Д
7Full_Model/batch_normalization/batchnorm/ReadVariableOpReadVariableOp@full_model_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0s
.Full_Model/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:д
,Full_Model/batch_normalization/batchnorm/addAddV2?Full_Model/batch_normalization/batchnorm/ReadVariableOp:value:07Full_Model/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:@
.Full_Model/batch_normalization/batchnorm/RsqrtRsqrt0Full_Model/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:@М
;Full_Model/batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOpDfull_model_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0б
,Full_Model/batch_normalization/batchnorm/mulMul2Full_Model/batch_normalization/batchnorm/Rsqrt:y:0CFull_Model/batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@С
.Full_Model/batch_normalization/batchnorm/mul_1Mul"Full_Model/conv1d/BiasAdd:output:00Full_Model/batch_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:џџџџџџџџџd@И
9Full_Model/batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOpBfull_model_batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0Я
.Full_Model/batch_normalization/batchnorm/mul_2MulAFull_Model/batch_normalization/batchnorm/ReadVariableOp_1:value:00Full_Model/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:@И
9Full_Model/batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOpBfull_model_batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0Я
,Full_Model/batch_normalization/batchnorm/subSubAFull_Model/batch_normalization/batchnorm/ReadVariableOp_2:value:02Full_Model/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@г
.Full_Model/batch_normalization/batchnorm/add_1AddV22Full_Model/batch_normalization/batchnorm/mul_1:z:00Full_Model/batch_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:џџџџџџџџџd@
Full_Model/re_lu_3/ReluRelu5Full_Model/batch_normalization_3/FusedBatchNormV3:y:0*
T0*5
_output_shapes#
!:џџџџџџџџџd@
Full_Model/re_lu/ReluRelu2Full_Model/batch_normalization/batchnorm/add_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџd@Ј
)Full_Model/conv3d_1/Conv3D/ReadVariableOpReadVariableOp2full_model_conv3d_1_conv3d_readvariableop_resource**
_output_shapes
:@@*
dtype0ч
Full_Model/conv3d_1/Conv3DConv3D%Full_Model/re_lu_3/Relu:activations:01Full_Model/conv3d_1/Conv3D/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:џџџџџџџџџd@*
paddingSAME*
strides	

*Full_Model/conv3d_1/BiasAdd/ReadVariableOpReadVariableOp3full_model_conv3d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0П
Full_Model/conv3d_1/BiasAddBiasAdd#Full_Model/conv3d_1/Conv3D:output:02Full_Model/conv3d_1/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:џџџџџџџџџd@t
)Full_Model/conv1d_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџЦ
%Full_Model/conv1d_1/Conv1D/ExpandDims
ExpandDims#Full_Model/re_lu/Relu:activations:02Full_Model/conv1d_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџd@К
6Full_Model/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp?full_model_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0m
+Full_Model/conv1d_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : м
'Full_Model/conv1d_1/Conv1D/ExpandDims_1
ExpandDims>Full_Model/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp:value:04Full_Model/conv1d_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@ш
Full_Model/conv1d_1/Conv1DConv2D.Full_Model/conv1d_1/Conv1D/ExpandDims:output:00Full_Model/conv1d_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџd@*
paddingSAME*
strides
Ј
"Full_Model/conv1d_1/Conv1D/SqueezeSqueeze#Full_Model/conv1d_1/Conv1D:output:0*
T0*+
_output_shapes
:џџџџџџџџџd@*
squeeze_dims

§џџџџџџџџ
*Full_Model/conv1d_1/BiasAdd/ReadVariableOpReadVariableOp3full_model_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Н
Full_Model/conv1d_1/BiasAddBiasAdd+Full_Model/conv1d_1/Conv1D/Squeeze:output:02Full_Model/conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџd@Є
/Full_Model/batch_normalization_4/ReadVariableOpReadVariableOp8full_model_batch_normalization_4_readvariableop_resource*
_output_shapes
:@*
dtype0Ј
1Full_Model/batch_normalization_4/ReadVariableOp_1ReadVariableOp:full_model_batch_normalization_4_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ц
@Full_Model/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpIfull_model_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0Ъ
BFull_Model/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpKfull_model_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0
1Full_Model/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3$Full_Model/conv3d_1/BiasAdd:output:07Full_Model/batch_normalization_4/ReadVariableOp:value:09Full_Model/batch_normalization_4/ReadVariableOp_1:value:0HFull_Model/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0JFull_Model/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*Q
_output_shapes?
=:џџџџџџџџџd@:@:@:@:@:*
data_formatNDHWC*
epsilon%o:*
is_training( И
9Full_Model/batch_normalization_1/batchnorm/ReadVariableOpReadVariableOpBfull_model_batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0u
0Full_Model/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:к
.Full_Model/batch_normalization_1/batchnorm/addAddV2AFull_Model/batch_normalization_1/batchnorm/ReadVariableOp:value:09Full_Model/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes
:@
0Full_Model/batch_normalization_1/batchnorm/RsqrtRsqrt2Full_Model/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
:@Р
=Full_Model/batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpFfull_model_batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0з
.Full_Model/batch_normalization_1/batchnorm/mulMul4Full_Model/batch_normalization_1/batchnorm/Rsqrt:y:0EFull_Model/batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@Ч
0Full_Model/batch_normalization_1/batchnorm/mul_1Mul$Full_Model/conv1d_1/BiasAdd:output:02Full_Model/batch_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:џџџџџџџџџd@М
;Full_Model/batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOpDfull_model_batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0е
0Full_Model/batch_normalization_1/batchnorm/mul_2MulCFull_Model/batch_normalization_1/batchnorm/ReadVariableOp_1:value:02Full_Model/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
:@М
;Full_Model/batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOpDfull_model_batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0е
.Full_Model/batch_normalization_1/batchnorm/subSubCFull_Model/batch_normalization_1/batchnorm/ReadVariableOp_2:value:04Full_Model/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@й
0Full_Model/batch_normalization_1/batchnorm/add_1AddV24Full_Model/batch_normalization_1/batchnorm/mul_1:z:02Full_Model/batch_normalization_1/batchnorm/sub:z:0*
T0*+
_output_shapes
:џџџџџџџџџd@
Full_Model/re_lu_4/ReluRelu5Full_Model/batch_normalization_4/FusedBatchNormV3:y:0*
T0*5
_output_shapes#
!:џџџџџџџџџd@
Full_Model/re_lu_1/ReluRelu4Full_Model/batch_normalization_1/batchnorm/add_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџd@Ј
)Full_Model/conv3d_2/Conv3D/ReadVariableOpReadVariableOp2full_model_conv3d_2_conv3d_readvariableop_resource**
_output_shapes
:@@*
dtype0ч
Full_Model/conv3d_2/Conv3DConv3D%Full_Model/re_lu_4/Relu:activations:01Full_Model/conv3d_2/Conv3D/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:џџџџџџџџџd@*
paddingSAME*
strides	

*Full_Model/conv3d_2/BiasAdd/ReadVariableOpReadVariableOp3full_model_conv3d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0П
Full_Model/conv3d_2/BiasAddBiasAdd#Full_Model/conv3d_2/Conv3D:output:02Full_Model/conv3d_2/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:џџџџџџџџџd@t
)Full_Model/conv1d_2/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџШ
%Full_Model/conv1d_2/Conv1D/ExpandDims
ExpandDims%Full_Model/re_lu_1/Relu:activations:02Full_Model/conv1d_2/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџd@К
6Full_Model/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp?full_model_conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0m
+Full_Model/conv1d_2/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : м
'Full_Model/conv1d_2/Conv1D/ExpandDims_1
ExpandDims>Full_Model/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp:value:04Full_Model/conv1d_2/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@ш
Full_Model/conv1d_2/Conv1DConv2D.Full_Model/conv1d_2/Conv1D/ExpandDims:output:00Full_Model/conv1d_2/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџd@*
paddingSAME*
strides
Ј
"Full_Model/conv1d_2/Conv1D/SqueezeSqueeze#Full_Model/conv1d_2/Conv1D:output:0*
T0*+
_output_shapes
:џџџџџџџџџd@*
squeeze_dims

§џџџџџџџџ
*Full_Model/conv1d_2/BiasAdd/ReadVariableOpReadVariableOp3full_model_conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Н
Full_Model/conv1d_2/BiasAddBiasAdd+Full_Model/conv1d_2/Conv1D/Squeeze:output:02Full_Model/conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџd@Є
/Full_Model/batch_normalization_5/ReadVariableOpReadVariableOp8full_model_batch_normalization_5_readvariableop_resource*
_output_shapes
:@*
dtype0Ј
1Full_Model/batch_normalization_5/ReadVariableOp_1ReadVariableOp:full_model_batch_normalization_5_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ц
@Full_Model/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpIfull_model_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0Ъ
BFull_Model/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpKfull_model_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0
1Full_Model/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3$Full_Model/conv3d_2/BiasAdd:output:07Full_Model/batch_normalization_5/ReadVariableOp:value:09Full_Model/batch_normalization_5/ReadVariableOp_1:value:0HFull_Model/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0JFull_Model/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*Q
_output_shapes?
=:џџџџџџџџџd@:@:@:@:@:*
data_formatNDHWC*
epsilon%o:*
is_training( И
9Full_Model/batch_normalization_2/batchnorm/ReadVariableOpReadVariableOpBfull_model_batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0u
0Full_Model/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:к
.Full_Model/batch_normalization_2/batchnorm/addAddV2AFull_Model/batch_normalization_2/batchnorm/ReadVariableOp:value:09Full_Model/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:@
0Full_Model/batch_normalization_2/batchnorm/RsqrtRsqrt2Full_Model/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:@Р
=Full_Model/batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpFfull_model_batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0з
.Full_Model/batch_normalization_2/batchnorm/mulMul4Full_Model/batch_normalization_2/batchnorm/Rsqrt:y:0EFull_Model/batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@Ч
0Full_Model/batch_normalization_2/batchnorm/mul_1Mul$Full_Model/conv1d_2/BiasAdd:output:02Full_Model/batch_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:џџџџџџџџџd@М
;Full_Model/batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOpDfull_model_batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0е
0Full_Model/batch_normalization_2/batchnorm/mul_2MulCFull_Model/batch_normalization_2/batchnorm/ReadVariableOp_1:value:02Full_Model/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:@М
;Full_Model/batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOpDfull_model_batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0е
.Full_Model/batch_normalization_2/batchnorm/subSubCFull_Model/batch_normalization_2/batchnorm/ReadVariableOp_2:value:04Full_Model/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@й
0Full_Model/batch_normalization_2/batchnorm/add_1AddV24Full_Model/batch_normalization_2/batchnorm/mul_1:z:02Full_Model/batch_normalization_2/batchnorm/sub:z:0*
T0*+
_output_shapes
:џџџџџџџџџd@
Full_Model/re_lu_5/ReluRelu5Full_Model/batch_normalization_5/FusedBatchNormV3:y:0*
T0*5
_output_shapes#
!:џџџџџџџџџd@
Full_Model/re_lu_2/ReluRelu4Full_Model/batch_normalization_2/batchnorm/add_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџd@
:Full_Model/global_average_pooling3d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         Ю
(Full_Model/global_average_pooling3d/MeanMean%Full_Model/re_lu_5/Relu:activations:0CFull_Model/global_average_pooling3d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@|
:Full_Model/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :Ю
(Full_Model/global_average_pooling1d/MeanMean%Full_Model/re_lu_2/Relu:activations:0CFull_Model/global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@
&Full_Model/dense/MatMul/ReadVariableOpReadVariableOp/full_model_dense_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Ж
Full_Model/dense/MatMulMatMul1Full_Model/global_average_pooling1d/Mean:output:0.Full_Model/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
'Full_Model/dense/BiasAdd/ReadVariableOpReadVariableOp0full_model_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Љ
Full_Model/dense/BiasAddBiasAdd!Full_Model/dense/MatMul:product:0/Full_Model/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџx
Full_Model/dense/SoftmaxSoftmax!Full_Model/dense/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
(Full_Model/dense_1/MatMul/ReadVariableOpReadVariableOp1full_model_dense_1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0К
Full_Model/dense_1/MatMulMatMul1Full_Model/global_average_pooling3d/Mean:output:00Full_Model/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
)Full_Model/dense_1/BiasAdd/ReadVariableOpReadVariableOp2full_model_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Џ
Full_Model/dense_1/BiasAddBiasAdd#Full_Model/dense_1/MatMul:product:01Full_Model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ|
Full_Model/dense_1/SoftmaxSoftmax#Full_Model/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџd
"Full_Model/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :л
Full_Model/concatenate/concatConcatV2"Full_Model/dense/Softmax:softmax:0$Full_Model/dense_1/Softmax:softmax:0+Full_Model/concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ

(Full_Model/dense_2/MatMul/ReadVariableOpReadVariableOp1full_model_dense_2_matmul_readvariableop_resource*
_output_shapes

:
 *
dtype0Џ
Full_Model/dense_2/MatMulMatMul&Full_Model/concatenate/concat:output:00Full_Model/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
)Full_Model/dense_2/BiasAdd/ReadVariableOpReadVariableOp2full_model_dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Џ
Full_Model/dense_2/BiasAddBiasAdd#Full_Model/dense_2/MatMul:product:01Full_Model/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
(Full_Model/dense_3/MatMul/ReadVariableOpReadVariableOp1full_model_dense_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype0Ќ
Full_Model/dense_3/MatMulMatMul#Full_Model/dense_2/BiasAdd:output:00Full_Model/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
)Full_Model/dense_3/BiasAdd/ReadVariableOpReadVariableOp2full_model_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Џ
Full_Model/dense_3/BiasAddBiasAdd#Full_Model/dense_3/MatMul:product:01Full_Model/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
(Full_Model/dense_4/MatMul/ReadVariableOpReadVariableOp1full_model_dense_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ќ
Full_Model/dense_4/MatMulMatMul#Full_Model/dense_3/BiasAdd:output:00Full_Model/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
)Full_Model/dense_4/BiasAdd/ReadVariableOpReadVariableOp2full_model_dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Џ
Full_Model/dense_4/BiasAddBiasAdd#Full_Model/dense_4/MatMul:product:01Full_Model/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
(Full_Model/dense_5/MatMul/ReadVariableOpReadVariableOp1full_model_dense_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ќ
Full_Model/dense_5/MatMulMatMul#Full_Model/dense_4/BiasAdd:output:00Full_Model/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
)Full_Model/dense_5/BiasAdd/ReadVariableOpReadVariableOp2full_model_dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Џ
Full_Model/dense_5/BiasAddBiasAdd#Full_Model/dense_5/MatMul:product:01Full_Model/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ|
Full_Model/dense_5/SoftmaxSoftmax#Full_Model/dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџs
IdentityIdentity$Full_Model/dense_5/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp8^Full_Model/batch_normalization/batchnorm/ReadVariableOp:^Full_Model/batch_normalization/batchnorm/ReadVariableOp_1:^Full_Model/batch_normalization/batchnorm/ReadVariableOp_2<^Full_Model/batch_normalization/batchnorm/mul/ReadVariableOp:^Full_Model/batch_normalization_1/batchnorm/ReadVariableOp<^Full_Model/batch_normalization_1/batchnorm/ReadVariableOp_1<^Full_Model/batch_normalization_1/batchnorm/ReadVariableOp_2>^Full_Model/batch_normalization_1/batchnorm/mul/ReadVariableOp:^Full_Model/batch_normalization_2/batchnorm/ReadVariableOp<^Full_Model/batch_normalization_2/batchnorm/ReadVariableOp_1<^Full_Model/batch_normalization_2/batchnorm/ReadVariableOp_2>^Full_Model/batch_normalization_2/batchnorm/mul/ReadVariableOpA^Full_Model/batch_normalization_3/FusedBatchNormV3/ReadVariableOpC^Full_Model/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_10^Full_Model/batch_normalization_3/ReadVariableOp2^Full_Model/batch_normalization_3/ReadVariableOp_1A^Full_Model/batch_normalization_4/FusedBatchNormV3/ReadVariableOpC^Full_Model/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_10^Full_Model/batch_normalization_4/ReadVariableOp2^Full_Model/batch_normalization_4/ReadVariableOp_1A^Full_Model/batch_normalization_5/FusedBatchNormV3/ReadVariableOpC^Full_Model/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_10^Full_Model/batch_normalization_5/ReadVariableOp2^Full_Model/batch_normalization_5/ReadVariableOp_1)^Full_Model/conv1d/BiasAdd/ReadVariableOp5^Full_Model/conv1d/Conv1D/ExpandDims_1/ReadVariableOp+^Full_Model/conv1d_1/BiasAdd/ReadVariableOp7^Full_Model/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp+^Full_Model/conv1d_2/BiasAdd/ReadVariableOp7^Full_Model/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp)^Full_Model/conv3d/BiasAdd/ReadVariableOp(^Full_Model/conv3d/Conv3D/ReadVariableOp+^Full_Model/conv3d_1/BiasAdd/ReadVariableOp*^Full_Model/conv3d_1/Conv3D/ReadVariableOp+^Full_Model/conv3d_2/BiasAdd/ReadVariableOp*^Full_Model/conv3d_2/Conv3D/ReadVariableOp(^Full_Model/dense/BiasAdd/ReadVariableOp'^Full_Model/dense/MatMul/ReadVariableOp*^Full_Model/dense_1/BiasAdd/ReadVariableOp)^Full_Model/dense_1/MatMul/ReadVariableOp*^Full_Model/dense_2/BiasAdd/ReadVariableOp)^Full_Model/dense_2/MatMul/ReadVariableOp*^Full_Model/dense_3/BiasAdd/ReadVariableOp)^Full_Model/dense_3/MatMul/ReadVariableOp*^Full_Model/dense_4/BiasAdd/ReadVariableOp)^Full_Model/dense_4/MatMul/ReadVariableOp*^Full_Model/dense_5/BiasAdd/ReadVariableOp)^Full_Model/dense_5/MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*­
_input_shapes
:џџџџџџџџџdt:џџџџџџџџџd: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2r
7Full_Model/batch_normalization/batchnorm/ReadVariableOp7Full_Model/batch_normalization/batchnorm/ReadVariableOp2v
9Full_Model/batch_normalization/batchnorm/ReadVariableOp_19Full_Model/batch_normalization/batchnorm/ReadVariableOp_12v
9Full_Model/batch_normalization/batchnorm/ReadVariableOp_29Full_Model/batch_normalization/batchnorm/ReadVariableOp_22z
;Full_Model/batch_normalization/batchnorm/mul/ReadVariableOp;Full_Model/batch_normalization/batchnorm/mul/ReadVariableOp2v
9Full_Model/batch_normalization_1/batchnorm/ReadVariableOp9Full_Model/batch_normalization_1/batchnorm/ReadVariableOp2z
;Full_Model/batch_normalization_1/batchnorm/ReadVariableOp_1;Full_Model/batch_normalization_1/batchnorm/ReadVariableOp_12z
;Full_Model/batch_normalization_1/batchnorm/ReadVariableOp_2;Full_Model/batch_normalization_1/batchnorm/ReadVariableOp_22~
=Full_Model/batch_normalization_1/batchnorm/mul/ReadVariableOp=Full_Model/batch_normalization_1/batchnorm/mul/ReadVariableOp2v
9Full_Model/batch_normalization_2/batchnorm/ReadVariableOp9Full_Model/batch_normalization_2/batchnorm/ReadVariableOp2z
;Full_Model/batch_normalization_2/batchnorm/ReadVariableOp_1;Full_Model/batch_normalization_2/batchnorm/ReadVariableOp_12z
;Full_Model/batch_normalization_2/batchnorm/ReadVariableOp_2;Full_Model/batch_normalization_2/batchnorm/ReadVariableOp_22~
=Full_Model/batch_normalization_2/batchnorm/mul/ReadVariableOp=Full_Model/batch_normalization_2/batchnorm/mul/ReadVariableOp2
@Full_Model/batch_normalization_3/FusedBatchNormV3/ReadVariableOp@Full_Model/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2
BFull_Model/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1BFull_Model/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12b
/Full_Model/batch_normalization_3/ReadVariableOp/Full_Model/batch_normalization_3/ReadVariableOp2f
1Full_Model/batch_normalization_3/ReadVariableOp_11Full_Model/batch_normalization_3/ReadVariableOp_12
@Full_Model/batch_normalization_4/FusedBatchNormV3/ReadVariableOp@Full_Model/batch_normalization_4/FusedBatchNormV3/ReadVariableOp2
BFull_Model/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1BFull_Model/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12b
/Full_Model/batch_normalization_4/ReadVariableOp/Full_Model/batch_normalization_4/ReadVariableOp2f
1Full_Model/batch_normalization_4/ReadVariableOp_11Full_Model/batch_normalization_4/ReadVariableOp_12
@Full_Model/batch_normalization_5/FusedBatchNormV3/ReadVariableOp@Full_Model/batch_normalization_5/FusedBatchNormV3/ReadVariableOp2
BFull_Model/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1BFull_Model/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12b
/Full_Model/batch_normalization_5/ReadVariableOp/Full_Model/batch_normalization_5/ReadVariableOp2f
1Full_Model/batch_normalization_5/ReadVariableOp_11Full_Model/batch_normalization_5/ReadVariableOp_12T
(Full_Model/conv1d/BiasAdd/ReadVariableOp(Full_Model/conv1d/BiasAdd/ReadVariableOp2l
4Full_Model/conv1d/Conv1D/ExpandDims_1/ReadVariableOp4Full_Model/conv1d/Conv1D/ExpandDims_1/ReadVariableOp2X
*Full_Model/conv1d_1/BiasAdd/ReadVariableOp*Full_Model/conv1d_1/BiasAdd/ReadVariableOp2p
6Full_Model/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp6Full_Model/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp2X
*Full_Model/conv1d_2/BiasAdd/ReadVariableOp*Full_Model/conv1d_2/BiasAdd/ReadVariableOp2p
6Full_Model/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp6Full_Model/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp2T
(Full_Model/conv3d/BiasAdd/ReadVariableOp(Full_Model/conv3d/BiasAdd/ReadVariableOp2R
'Full_Model/conv3d/Conv3D/ReadVariableOp'Full_Model/conv3d/Conv3D/ReadVariableOp2X
*Full_Model/conv3d_1/BiasAdd/ReadVariableOp*Full_Model/conv3d_1/BiasAdd/ReadVariableOp2V
)Full_Model/conv3d_1/Conv3D/ReadVariableOp)Full_Model/conv3d_1/Conv3D/ReadVariableOp2X
*Full_Model/conv3d_2/BiasAdd/ReadVariableOp*Full_Model/conv3d_2/BiasAdd/ReadVariableOp2V
)Full_Model/conv3d_2/Conv3D/ReadVariableOp)Full_Model/conv3d_2/Conv3D/ReadVariableOp2R
'Full_Model/dense/BiasAdd/ReadVariableOp'Full_Model/dense/BiasAdd/ReadVariableOp2P
&Full_Model/dense/MatMul/ReadVariableOp&Full_Model/dense/MatMul/ReadVariableOp2V
)Full_Model/dense_1/BiasAdd/ReadVariableOp)Full_Model/dense_1/BiasAdd/ReadVariableOp2T
(Full_Model/dense_1/MatMul/ReadVariableOp(Full_Model/dense_1/MatMul/ReadVariableOp2V
)Full_Model/dense_2/BiasAdd/ReadVariableOp)Full_Model/dense_2/BiasAdd/ReadVariableOp2T
(Full_Model/dense_2/MatMul/ReadVariableOp(Full_Model/dense_2/MatMul/ReadVariableOp2V
)Full_Model/dense_3/BiasAdd/ReadVariableOp)Full_Model/dense_3/BiasAdd/ReadVariableOp2T
(Full_Model/dense_3/MatMul/ReadVariableOp(Full_Model/dense_3/MatMul/ReadVariableOp2V
)Full_Model/dense_4/BiasAdd/ReadVariableOp)Full_Model/dense_4/BiasAdd/ReadVariableOp2T
(Full_Model/dense_4/MatMul/ReadVariableOp(Full_Model/dense_4/MatMul/ReadVariableOp2V
)Full_Model/dense_5/BiasAdd/ReadVariableOp)Full_Model/dense_5/BiasAdd/ReadVariableOp2T
(Full_Model/dense_5/MatMul/ReadVariableOp(Full_Model/dense_5/MatMul/ReadVariableOp:T P
+
_output_shapes
:џџџџџџџџџdt
!
_user_specified_name	input_1:^Z
5
_output_shapes#
!:џџџџџџџџџd
!
_user_specified_name	input_2:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:( $
"
_user_specified_name
resource:(!$
"
_user_specified_name
resource:("$
"
_user_specified_name
resource:(#$
"
_user_specified_name
resource:($$
"
_user_specified_name
resource:(%$
"
_user_specified_name
resource:(&$
"
_user_specified_name
resource:('$
"
_user_specified_name
resource:(($
"
_user_specified_name
resource:()$
"
_user_specified_name
resource:(*$
"
_user_specified_name
resource:(+$
"
_user_specified_name
resource:(,$
"
_user_specified_name
resource:(-$
"
_user_specified_name
resource:(.$
"
_user_specified_name
resource:(/$
"
_user_specified_name
resource:(0$
"
_user_specified_name
resource:(1$
"
_user_specified_name
resource
е
О
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_2778

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0ћ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*j
_output_shapesX
V:8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@:@:@:@:@:*
data_formatNDHWC*
epsilon%o:*
exponential_avg_factor%
з#<Ц
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(а
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*N
_output_shapes<
::8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@А
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:v r
N
_output_shapes<
::8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource


B__inference_conv1d_2_layer_call_and_return_conditional_losses_2635

inputsA
+conv1d_expanddims_1_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂ"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџd@
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :  
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@Ќ
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџd@*
paddingSAME*
strides

Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:џџџџџџџџџd@*
squeeze_dims

§џџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџd@c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџd@`
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџd@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:џџџџџџџџџd@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
щ

&__inference_dense_1_layer_call_fn_2867

inputs
unknown:@
	unknown_0:
identityЂStatefulPartitionedCallл
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_1544o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs:$ 

_user_specified_name2861:$ 

_user_specified_name2863

n
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_2827

inputs
identityX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Я	
Я
4__inference_batch_normalization_2_layer_call_fn_2667

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1187|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:џџџџџџџџџџџџџџџџџџ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs:$ 

_user_specified_name2657:$ 

_user_specified_name2659:$ 

_user_specified_name2661:$ 

_user_specified_name2663
ю
к1
__inference__traced_save_3315
file_prefix:
$read_disablecopyonread_conv1d_kernel:t@2
$read_1_disablecopyonread_conv1d_bias:@D
&read_2_disablecopyonread_conv3d_kernel:@2
$read_3_disablecopyonread_conv3d_bias:@@
2read_4_disablecopyonread_batch_normalization_gamma:@?
1read_5_disablecopyonread_batch_normalization_beta:@F
8read_6_disablecopyonread_batch_normalization_moving_mean:@J
<read_7_disablecopyonread_batch_normalization_moving_variance:@B
4read_8_disablecopyonread_batch_normalization_3_gamma:@A
3read_9_disablecopyonread_batch_normalization_3_beta:@I
;read_10_disablecopyonread_batch_normalization_3_moving_mean:@M
?read_11_disablecopyonread_batch_normalization_3_moving_variance:@?
)read_12_disablecopyonread_conv1d_1_kernel:@@5
'read_13_disablecopyonread_conv1d_1_bias:@G
)read_14_disablecopyonread_conv3d_1_kernel:@@5
'read_15_disablecopyonread_conv3d_1_bias:@C
5read_16_disablecopyonread_batch_normalization_1_gamma:@B
4read_17_disablecopyonread_batch_normalization_1_beta:@I
;read_18_disablecopyonread_batch_normalization_1_moving_mean:@M
?read_19_disablecopyonread_batch_normalization_1_moving_variance:@C
5read_20_disablecopyonread_batch_normalization_4_gamma:@B
4read_21_disablecopyonread_batch_normalization_4_beta:@I
;read_22_disablecopyonread_batch_normalization_4_moving_mean:@M
?read_23_disablecopyonread_batch_normalization_4_moving_variance:@?
)read_24_disablecopyonread_conv1d_2_kernel:@@5
'read_25_disablecopyonread_conv1d_2_bias:@G
)read_26_disablecopyonread_conv3d_2_kernel:@@5
'read_27_disablecopyonread_conv3d_2_bias:@C
5read_28_disablecopyonread_batch_normalization_2_gamma:@B
4read_29_disablecopyonread_batch_normalization_2_beta:@I
;read_30_disablecopyonread_batch_normalization_2_moving_mean:@M
?read_31_disablecopyonread_batch_normalization_2_moving_variance:@C
5read_32_disablecopyonread_batch_normalization_5_gamma:@B
4read_33_disablecopyonread_batch_normalization_5_beta:@I
;read_34_disablecopyonread_batch_normalization_5_moving_mean:@M
?read_35_disablecopyonread_batch_normalization_5_moving_variance:@8
&read_36_disablecopyonread_dense_kernel:@2
$read_37_disablecopyonread_dense_bias::
(read_38_disablecopyonread_dense_1_kernel:@4
&read_39_disablecopyonread_dense_1_bias::
(read_40_disablecopyonread_dense_2_kernel:
 4
&read_41_disablecopyonread_dense_2_bias: :
(read_42_disablecopyonread_dense_3_kernel: 4
&read_43_disablecopyonread_dense_3_bias::
(read_44_disablecopyonread_dense_4_kernel:4
&read_45_disablecopyonread_dense_4_bias::
(read_46_disablecopyonread_dense_5_kernel:4
&read_47_disablecopyonread_dense_5_bias:-
#read_48_disablecopyonread_iteration:	 1
'read_49_disablecopyonread_learning_rate: +
!read_50_disablecopyonread_total_1: +
!read_51_disablecopyonread_count_1: )
read_52_disablecopyonread_total: )
read_53_disablecopyonread_count: 
savev2_const
identity_109ЂMergeV2CheckpointsЂRead/DisableCopyOnReadЂRead/ReadVariableOpЂRead_1/DisableCopyOnReadЂRead_1/ReadVariableOpЂRead_10/DisableCopyOnReadЂRead_10/ReadVariableOpЂRead_11/DisableCopyOnReadЂRead_11/ReadVariableOpЂRead_12/DisableCopyOnReadЂRead_12/ReadVariableOpЂRead_13/DisableCopyOnReadЂRead_13/ReadVariableOpЂRead_14/DisableCopyOnReadЂRead_14/ReadVariableOpЂRead_15/DisableCopyOnReadЂRead_15/ReadVariableOpЂRead_16/DisableCopyOnReadЂRead_16/ReadVariableOpЂRead_17/DisableCopyOnReadЂRead_17/ReadVariableOpЂRead_18/DisableCopyOnReadЂRead_18/ReadVariableOpЂRead_19/DisableCopyOnReadЂRead_19/ReadVariableOpЂRead_2/DisableCopyOnReadЂRead_2/ReadVariableOpЂRead_20/DisableCopyOnReadЂRead_20/ReadVariableOpЂRead_21/DisableCopyOnReadЂRead_21/ReadVariableOpЂRead_22/DisableCopyOnReadЂRead_22/ReadVariableOpЂRead_23/DisableCopyOnReadЂRead_23/ReadVariableOpЂRead_24/DisableCopyOnReadЂRead_24/ReadVariableOpЂRead_25/DisableCopyOnReadЂRead_25/ReadVariableOpЂRead_26/DisableCopyOnReadЂRead_26/ReadVariableOpЂRead_27/DisableCopyOnReadЂRead_27/ReadVariableOpЂRead_28/DisableCopyOnReadЂRead_28/ReadVariableOpЂRead_29/DisableCopyOnReadЂRead_29/ReadVariableOpЂRead_3/DisableCopyOnReadЂRead_3/ReadVariableOpЂRead_30/DisableCopyOnReadЂRead_30/ReadVariableOpЂRead_31/DisableCopyOnReadЂRead_31/ReadVariableOpЂRead_32/DisableCopyOnReadЂRead_32/ReadVariableOpЂRead_33/DisableCopyOnReadЂRead_33/ReadVariableOpЂRead_34/DisableCopyOnReadЂRead_34/ReadVariableOpЂRead_35/DisableCopyOnReadЂRead_35/ReadVariableOpЂRead_36/DisableCopyOnReadЂRead_36/ReadVariableOpЂRead_37/DisableCopyOnReadЂRead_37/ReadVariableOpЂRead_38/DisableCopyOnReadЂRead_38/ReadVariableOpЂRead_39/DisableCopyOnReadЂRead_39/ReadVariableOpЂRead_4/DisableCopyOnReadЂRead_4/ReadVariableOpЂRead_40/DisableCopyOnReadЂRead_40/ReadVariableOpЂRead_41/DisableCopyOnReadЂRead_41/ReadVariableOpЂRead_42/DisableCopyOnReadЂRead_42/ReadVariableOpЂRead_43/DisableCopyOnReadЂRead_43/ReadVariableOpЂRead_44/DisableCopyOnReadЂRead_44/ReadVariableOpЂRead_45/DisableCopyOnReadЂRead_45/ReadVariableOpЂRead_46/DisableCopyOnReadЂRead_46/ReadVariableOpЂRead_47/DisableCopyOnReadЂRead_47/ReadVariableOpЂRead_48/DisableCopyOnReadЂRead_48/ReadVariableOpЂRead_49/DisableCopyOnReadЂRead_49/ReadVariableOpЂRead_5/DisableCopyOnReadЂRead_5/ReadVariableOpЂRead_50/DisableCopyOnReadЂRead_50/ReadVariableOpЂRead_51/DisableCopyOnReadЂRead_51/ReadVariableOpЂRead_52/DisableCopyOnReadЂRead_52/ReadVariableOpЂRead_53/DisableCopyOnReadЂRead_53/ReadVariableOpЂRead_6/DisableCopyOnReadЂRead_6/ReadVariableOpЂRead_7/DisableCopyOnReadЂRead_7/ReadVariableOpЂRead_8/DisableCopyOnReadЂRead_8/ReadVariableOpЂRead_9/DisableCopyOnReadЂRead_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: v
Read/DisableCopyOnReadDisableCopyOnRead$read_disablecopyonread_conv1d_kernel"/device:CPU:0*
_output_shapes
 Є
Read/ReadVariableOpReadVariableOp$read_disablecopyonread_conv1d_kernel^Read/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:t@*
dtype0m
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:t@e

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*"
_output_shapes
:t@x
Read_1/DisableCopyOnReadDisableCopyOnRead$read_1_disablecopyonread_conv1d_bias"/device:CPU:0*
_output_shapes
  
Read_1/ReadVariableOpReadVariableOp$read_1_disablecopyonread_conv1d_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:@z
Read_2/DisableCopyOnReadDisableCopyOnRead&read_2_disablecopyonread_conv3d_kernel"/device:CPU:0*
_output_shapes
 В
Read_2/ReadVariableOpReadVariableOp&read_2_disablecopyonread_conv3d_kernel^Read_2/DisableCopyOnRead"/device:CPU:0**
_output_shapes
:@*
dtype0y

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0**
_output_shapes
:@o

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0**
_output_shapes
:@x
Read_3/DisableCopyOnReadDisableCopyOnRead$read_3_disablecopyonread_conv3d_bias"/device:CPU:0*
_output_shapes
  
Read_3/ReadVariableOpReadVariableOp$read_3_disablecopyonread_conv3d_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@_

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:@
Read_4/DisableCopyOnReadDisableCopyOnRead2read_4_disablecopyonread_batch_normalization_gamma"/device:CPU:0*
_output_shapes
 Ў
Read_4/ReadVariableOpReadVariableOp2read_4_disablecopyonread_batch_normalization_gamma^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0i

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@_

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes
:@
Read_5/DisableCopyOnReadDisableCopyOnRead1read_5_disablecopyonread_batch_normalization_beta"/device:CPU:0*
_output_shapes
 ­
Read_5/ReadVariableOpReadVariableOp1read_5_disablecopyonread_batch_normalization_beta^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0j
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
:@
Read_6/DisableCopyOnReadDisableCopyOnRead8read_6_disablecopyonread_batch_normalization_moving_mean"/device:CPU:0*
_output_shapes
 Д
Read_6/ReadVariableOpReadVariableOp8read_6_disablecopyonread_batch_normalization_moving_mean^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0j
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes
:@
Read_7/DisableCopyOnReadDisableCopyOnRead<read_7_disablecopyonread_batch_normalization_moving_variance"/device:CPU:0*
_output_shapes
 И
Read_7/ReadVariableOpReadVariableOp<read_7_disablecopyonread_batch_normalization_moving_variance^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0j
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
:@
Read_8/DisableCopyOnReadDisableCopyOnRead4read_8_disablecopyonread_batch_normalization_3_gamma"/device:CPU:0*
_output_shapes
 А
Read_8/ReadVariableOpReadVariableOp4read_8_disablecopyonread_batch_normalization_3_gamma^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0j
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes
:@
Read_9/DisableCopyOnReadDisableCopyOnRead3read_9_disablecopyonread_batch_normalization_3_beta"/device:CPU:0*
_output_shapes
 Џ
Read_9/ReadVariableOpReadVariableOp3read_9_disablecopyonread_batch_normalization_3_beta^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0j
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
:@
Read_10/DisableCopyOnReadDisableCopyOnRead;read_10_disablecopyonread_batch_normalization_3_moving_mean"/device:CPU:0*
_output_shapes
 Й
Read_10/ReadVariableOpReadVariableOp;read_10_disablecopyonread_batch_normalization_3_moving_mean^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes
:@
Read_11/DisableCopyOnReadDisableCopyOnRead?read_11_disablecopyonread_batch_normalization_3_moving_variance"/device:CPU:0*
_output_shapes
 Н
Read_11/ReadVariableOpReadVariableOp?read_11_disablecopyonread_batch_normalization_3_moving_variance^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
:@~
Read_12/DisableCopyOnReadDisableCopyOnRead)read_12_disablecopyonread_conv1d_1_kernel"/device:CPU:0*
_output_shapes
 Џ
Read_12/ReadVariableOpReadVariableOp)read_12_disablecopyonread_conv1d_1_kernel^Read_12/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:@@*
dtype0s
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:@@i
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*"
_output_shapes
:@@|
Read_13/DisableCopyOnReadDisableCopyOnRead'read_13_disablecopyonread_conv1d_1_bias"/device:CPU:0*
_output_shapes
 Ѕ
Read_13/ReadVariableOpReadVariableOp'read_13_disablecopyonread_conv1d_1_bias^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
:@~
Read_14/DisableCopyOnReadDisableCopyOnRead)read_14_disablecopyonread_conv3d_1_kernel"/device:CPU:0*
_output_shapes
 З
Read_14/ReadVariableOpReadVariableOp)read_14_disablecopyonread_conv3d_1_kernel^Read_14/DisableCopyOnRead"/device:CPU:0**
_output_shapes
:@@*
dtype0{
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0**
_output_shapes
:@@q
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0**
_output_shapes
:@@|
Read_15/DisableCopyOnReadDisableCopyOnRead'read_15_disablecopyonread_conv3d_1_bias"/device:CPU:0*
_output_shapes
 Ѕ
Read_15/ReadVariableOpReadVariableOp'read_15_disablecopyonread_conv3d_1_bias^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
:@
Read_16/DisableCopyOnReadDisableCopyOnRead5read_16_disablecopyonread_batch_normalization_1_gamma"/device:CPU:0*
_output_shapes
 Г
Read_16/ReadVariableOpReadVariableOp5read_16_disablecopyonread_batch_normalization_1_gamma^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes
:@
Read_17/DisableCopyOnReadDisableCopyOnRead4read_17_disablecopyonread_batch_normalization_1_beta"/device:CPU:0*
_output_shapes
 В
Read_17/ReadVariableOpReadVariableOp4read_17_disablecopyonread_batch_normalization_1_beta^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
:@
Read_18/DisableCopyOnReadDisableCopyOnRead;read_18_disablecopyonread_batch_normalization_1_moving_mean"/device:CPU:0*
_output_shapes
 Й
Read_18/ReadVariableOpReadVariableOp;read_18_disablecopyonread_batch_normalization_1_moving_mean^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes
:@
Read_19/DisableCopyOnReadDisableCopyOnRead?read_19_disablecopyonread_batch_normalization_1_moving_variance"/device:CPU:0*
_output_shapes
 Н
Read_19/ReadVariableOpReadVariableOp?read_19_disablecopyonread_batch_normalization_1_moving_variance^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes
:@
Read_20/DisableCopyOnReadDisableCopyOnRead5read_20_disablecopyonread_batch_normalization_4_gamma"/device:CPU:0*
_output_shapes
 Г
Read_20/ReadVariableOpReadVariableOp5read_20_disablecopyonread_batch_normalization_4_gamma^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes
:@
Read_21/DisableCopyOnReadDisableCopyOnRead4read_21_disablecopyonread_batch_normalization_4_beta"/device:CPU:0*
_output_shapes
 В
Read_21/ReadVariableOpReadVariableOp4read_21_disablecopyonread_batch_normalization_4_beta^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes
:@
Read_22/DisableCopyOnReadDisableCopyOnRead;read_22_disablecopyonread_batch_normalization_4_moving_mean"/device:CPU:0*
_output_shapes
 Й
Read_22/ReadVariableOpReadVariableOp;read_22_disablecopyonread_batch_normalization_4_moving_mean^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes
:@
Read_23/DisableCopyOnReadDisableCopyOnRead?read_23_disablecopyonread_batch_normalization_4_moving_variance"/device:CPU:0*
_output_shapes
 Н
Read_23/ReadVariableOpReadVariableOp?read_23_disablecopyonread_batch_normalization_4_moving_variance^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes
:@~
Read_24/DisableCopyOnReadDisableCopyOnRead)read_24_disablecopyonread_conv1d_2_kernel"/device:CPU:0*
_output_shapes
 Џ
Read_24/ReadVariableOpReadVariableOp)read_24_disablecopyonread_conv1d_2_kernel^Read_24/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:@@*
dtype0s
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:@@i
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*"
_output_shapes
:@@|
Read_25/DisableCopyOnReadDisableCopyOnRead'read_25_disablecopyonread_conv1d_2_bias"/device:CPU:0*
_output_shapes
 Ѕ
Read_25/ReadVariableOpReadVariableOp'read_25_disablecopyonread_conv1d_2_bias^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes
:@~
Read_26/DisableCopyOnReadDisableCopyOnRead)read_26_disablecopyonread_conv3d_2_kernel"/device:CPU:0*
_output_shapes
 З
Read_26/ReadVariableOpReadVariableOp)read_26_disablecopyonread_conv3d_2_kernel^Read_26/DisableCopyOnRead"/device:CPU:0**
_output_shapes
:@@*
dtype0{
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0**
_output_shapes
:@@q
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0**
_output_shapes
:@@|
Read_27/DisableCopyOnReadDisableCopyOnRead'read_27_disablecopyonread_conv3d_2_bias"/device:CPU:0*
_output_shapes
 Ѕ
Read_27/ReadVariableOpReadVariableOp'read_27_disablecopyonread_conv3d_2_bias^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes
:@
Read_28/DisableCopyOnReadDisableCopyOnRead5read_28_disablecopyonread_batch_normalization_2_gamma"/device:CPU:0*
_output_shapes
 Г
Read_28/ReadVariableOpReadVariableOp5read_28_disablecopyonread_batch_normalization_2_gamma^Read_28/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*
_output_shapes
:@
Read_29/DisableCopyOnReadDisableCopyOnRead4read_29_disablecopyonread_batch_normalization_2_beta"/device:CPU:0*
_output_shapes
 В
Read_29/ReadVariableOpReadVariableOp4read_29_disablecopyonread_batch_normalization_2_beta^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes
:@
Read_30/DisableCopyOnReadDisableCopyOnRead;read_30_disablecopyonread_batch_normalization_2_moving_mean"/device:CPU:0*
_output_shapes
 Й
Read_30/ReadVariableOpReadVariableOp;read_30_disablecopyonread_batch_normalization_2_moving_mean^Read_30/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*
_output_shapes
:@
Read_31/DisableCopyOnReadDisableCopyOnRead?read_31_disablecopyonread_batch_normalization_2_moving_variance"/device:CPU:0*
_output_shapes
 Н
Read_31/ReadVariableOpReadVariableOp?read_31_disablecopyonread_batch_normalization_2_moving_variance^Read_31/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0*
_output_shapes
:@
Read_32/DisableCopyOnReadDisableCopyOnRead5read_32_disablecopyonread_batch_normalization_5_gamma"/device:CPU:0*
_output_shapes
 Г
Read_32/ReadVariableOpReadVariableOp5read_32_disablecopyonread_batch_normalization_5_gamma^Read_32/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_64IdentityRead_32/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*
_output_shapes
:@
Read_33/DisableCopyOnReadDisableCopyOnRead4read_33_disablecopyonread_batch_normalization_5_beta"/device:CPU:0*
_output_shapes
 В
Read_33/ReadVariableOpReadVariableOp4read_33_disablecopyonread_batch_normalization_5_beta^Read_33/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_66IdentityRead_33/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0*
_output_shapes
:@
Read_34/DisableCopyOnReadDisableCopyOnRead;read_34_disablecopyonread_batch_normalization_5_moving_mean"/device:CPU:0*
_output_shapes
 Й
Read_34/ReadVariableOpReadVariableOp;read_34_disablecopyonread_batch_normalization_5_moving_mean^Read_34/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_68IdentityRead_34/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0*
_output_shapes
:@
Read_35/DisableCopyOnReadDisableCopyOnRead?read_35_disablecopyonread_batch_normalization_5_moving_variance"/device:CPU:0*
_output_shapes
 Н
Read_35/ReadVariableOpReadVariableOp?read_35_disablecopyonread_batch_normalization_5_moving_variance^Read_35/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_70IdentityRead_35/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_71IdentityIdentity_70:output:0"/device:CPU:0*
T0*
_output_shapes
:@{
Read_36/DisableCopyOnReadDisableCopyOnRead&read_36_disablecopyonread_dense_kernel"/device:CPU:0*
_output_shapes
 Ј
Read_36/ReadVariableOpReadVariableOp&read_36_disablecopyonread_dense_kernel^Read_36/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_72IdentityRead_36/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_73IdentityIdentity_72:output:0"/device:CPU:0*
T0*
_output_shapes

:@y
Read_37/DisableCopyOnReadDisableCopyOnRead$read_37_disablecopyonread_dense_bias"/device:CPU:0*
_output_shapes
 Ђ
Read_37/ReadVariableOpReadVariableOp$read_37_disablecopyonread_dense_bias^Read_37/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_74IdentityRead_37/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_75IdentityIdentity_74:output:0"/device:CPU:0*
T0*
_output_shapes
:}
Read_38/DisableCopyOnReadDisableCopyOnRead(read_38_disablecopyonread_dense_1_kernel"/device:CPU:0*
_output_shapes
 Њ
Read_38/ReadVariableOpReadVariableOp(read_38_disablecopyonread_dense_1_kernel^Read_38/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_76IdentityRead_38/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_77IdentityIdentity_76:output:0"/device:CPU:0*
T0*
_output_shapes

:@{
Read_39/DisableCopyOnReadDisableCopyOnRead&read_39_disablecopyonread_dense_1_bias"/device:CPU:0*
_output_shapes
 Є
Read_39/ReadVariableOpReadVariableOp&read_39_disablecopyonread_dense_1_bias^Read_39/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_78IdentityRead_39/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_79IdentityIdentity_78:output:0"/device:CPU:0*
T0*
_output_shapes
:}
Read_40/DisableCopyOnReadDisableCopyOnRead(read_40_disablecopyonread_dense_2_kernel"/device:CPU:0*
_output_shapes
 Њ
Read_40/ReadVariableOpReadVariableOp(read_40_disablecopyonread_dense_2_kernel^Read_40/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:
 *
dtype0o
Identity_80IdentityRead_40/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:
 e
Identity_81IdentityIdentity_80:output:0"/device:CPU:0*
T0*
_output_shapes

:
 {
Read_41/DisableCopyOnReadDisableCopyOnRead&read_41_disablecopyonread_dense_2_bias"/device:CPU:0*
_output_shapes
 Є
Read_41/ReadVariableOpReadVariableOp&read_41_disablecopyonread_dense_2_bias^Read_41/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_82IdentityRead_41/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_83IdentityIdentity_82:output:0"/device:CPU:0*
T0*
_output_shapes
: }
Read_42/DisableCopyOnReadDisableCopyOnRead(read_42_disablecopyonread_dense_3_kernel"/device:CPU:0*
_output_shapes
 Њ
Read_42/ReadVariableOpReadVariableOp(read_42_disablecopyonread_dense_3_kernel^Read_42/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0o
Identity_84IdentityRead_42/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: e
Identity_85IdentityIdentity_84:output:0"/device:CPU:0*
T0*
_output_shapes

: {
Read_43/DisableCopyOnReadDisableCopyOnRead&read_43_disablecopyonread_dense_3_bias"/device:CPU:0*
_output_shapes
 Є
Read_43/ReadVariableOpReadVariableOp&read_43_disablecopyonread_dense_3_bias^Read_43/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_86IdentityRead_43/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_87IdentityIdentity_86:output:0"/device:CPU:0*
T0*
_output_shapes
:}
Read_44/DisableCopyOnReadDisableCopyOnRead(read_44_disablecopyonread_dense_4_kernel"/device:CPU:0*
_output_shapes
 Њ
Read_44/ReadVariableOpReadVariableOp(read_44_disablecopyonread_dense_4_kernel^Read_44/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_88IdentityRead_44/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_89IdentityIdentity_88:output:0"/device:CPU:0*
T0*
_output_shapes

:{
Read_45/DisableCopyOnReadDisableCopyOnRead&read_45_disablecopyonread_dense_4_bias"/device:CPU:0*
_output_shapes
 Є
Read_45/ReadVariableOpReadVariableOp&read_45_disablecopyonread_dense_4_bias^Read_45/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_90IdentityRead_45/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_91IdentityIdentity_90:output:0"/device:CPU:0*
T0*
_output_shapes
:}
Read_46/DisableCopyOnReadDisableCopyOnRead(read_46_disablecopyonread_dense_5_kernel"/device:CPU:0*
_output_shapes
 Њ
Read_46/ReadVariableOpReadVariableOp(read_46_disablecopyonread_dense_5_kernel^Read_46/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_92IdentityRead_46/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_93IdentityIdentity_92:output:0"/device:CPU:0*
T0*
_output_shapes

:{
Read_47/DisableCopyOnReadDisableCopyOnRead&read_47_disablecopyonread_dense_5_bias"/device:CPU:0*
_output_shapes
 Є
Read_47/ReadVariableOpReadVariableOp&read_47_disablecopyonread_dense_5_bias^Read_47/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_94IdentityRead_47/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_95IdentityIdentity_94:output:0"/device:CPU:0*
T0*
_output_shapes
:x
Read_48/DisableCopyOnReadDisableCopyOnRead#read_48_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 
Read_48/ReadVariableOpReadVariableOp#read_48_disablecopyonread_iteration^Read_48/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	g
Identity_96IdentityRead_48/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_97IdentityIdentity_96:output:0"/device:CPU:0*
T0	*
_output_shapes
: |
Read_49/DisableCopyOnReadDisableCopyOnRead'read_49_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 Ё
Read_49/ReadVariableOpReadVariableOp'read_49_disablecopyonread_learning_rate^Read_49/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_98IdentityRead_49/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_99IdentityIdentity_98:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_50/DisableCopyOnReadDisableCopyOnRead!read_50_disablecopyonread_total_1"/device:CPU:0*
_output_shapes
 
Read_50/ReadVariableOpReadVariableOp!read_50_disablecopyonread_total_1^Read_50/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_100IdentityRead_50/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_101IdentityIdentity_100:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_51/DisableCopyOnReadDisableCopyOnRead!read_51_disablecopyonread_count_1"/device:CPU:0*
_output_shapes
 
Read_51/ReadVariableOpReadVariableOp!read_51_disablecopyonread_count_1^Read_51/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_102IdentityRead_51/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_103IdentityIdentity_102:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_52/DisableCopyOnReadDisableCopyOnReadread_52_disablecopyonread_total"/device:CPU:0*
_output_shapes
 
Read_52/ReadVariableOpReadVariableOpread_52_disablecopyonread_total^Read_52/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_104IdentityRead_52/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_105IdentityIdentity_104:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_53/DisableCopyOnReadDisableCopyOnReadread_53_disablecopyonread_count"/device:CPU:0*
_output_shapes
 
Read_53/ReadVariableOpReadVariableOpread_53_disablecopyonread_count^Read_53/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_106IdentityRead_53/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_107IdentityIdentity_106:output:0"/device:CPU:0*
T0*
_output_shapes
: 
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*А
valueІBЃ7B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHм
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*
valuexBv7B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Б
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0Identity_73:output:0Identity_75:output:0Identity_77:output:0Identity_79:output:0Identity_81:output:0Identity_83:output:0Identity_85:output:0Identity_87:output:0Identity_89:output:0Identity_91:output:0Identity_93:output:0Identity_95:output:0Identity_97:output:0Identity_99:output:0Identity_101:output:0Identity_103:output:0Identity_105:output:0Identity_107:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *E
dtypes;
927	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Г
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 j
Identity_108Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: W
Identity_109IdentityIdentity_108:output:0^NoOp*
T0*
_output_shapes
: Э
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_36/DisableCopyOnRead^Read_36/ReadVariableOp^Read_37/DisableCopyOnRead^Read_37/ReadVariableOp^Read_38/DisableCopyOnRead^Read_38/ReadVariableOp^Read_39/DisableCopyOnRead^Read_39/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_40/DisableCopyOnRead^Read_40/ReadVariableOp^Read_41/DisableCopyOnRead^Read_41/ReadVariableOp^Read_42/DisableCopyOnRead^Read_42/ReadVariableOp^Read_43/DisableCopyOnRead^Read_43/ReadVariableOp^Read_44/DisableCopyOnRead^Read_44/ReadVariableOp^Read_45/DisableCopyOnRead^Read_45/ReadVariableOp^Read_46/DisableCopyOnRead^Read_46/ReadVariableOp^Read_47/DisableCopyOnRead^Read_47/ReadVariableOp^Read_48/DisableCopyOnRead^Read_48/ReadVariableOp^Read_49/DisableCopyOnRead^Read_49/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_50/DisableCopyOnRead^Read_50/ReadVariableOp^Read_51/DisableCopyOnRead^Read_51/ReadVariableOp^Read_52/DisableCopyOnRead^Read_52/ReadVariableOp^Read_53/DisableCopyOnRead^Read_53/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "%
identity_109Identity_109:output:0*(
_construction_contextkEagerRuntime*
_input_shapesr
p: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp26
Read_31/DisableCopyOnReadRead_31/DisableCopyOnRead20
Read_31/ReadVariableOpRead_31/ReadVariableOp26
Read_32/DisableCopyOnReadRead_32/DisableCopyOnRead20
Read_32/ReadVariableOpRead_32/ReadVariableOp26
Read_33/DisableCopyOnReadRead_33/DisableCopyOnRead20
Read_33/ReadVariableOpRead_33/ReadVariableOp26
Read_34/DisableCopyOnReadRead_34/DisableCopyOnRead20
Read_34/ReadVariableOpRead_34/ReadVariableOp26
Read_35/DisableCopyOnReadRead_35/DisableCopyOnRead20
Read_35/ReadVariableOpRead_35/ReadVariableOp26
Read_36/DisableCopyOnReadRead_36/DisableCopyOnRead20
Read_36/ReadVariableOpRead_36/ReadVariableOp26
Read_37/DisableCopyOnReadRead_37/DisableCopyOnRead20
Read_37/ReadVariableOpRead_37/ReadVariableOp26
Read_38/DisableCopyOnReadRead_38/DisableCopyOnRead20
Read_38/ReadVariableOpRead_38/ReadVariableOp26
Read_39/DisableCopyOnReadRead_39/DisableCopyOnRead20
Read_39/ReadVariableOpRead_39/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp26
Read_40/DisableCopyOnReadRead_40/DisableCopyOnRead20
Read_40/ReadVariableOpRead_40/ReadVariableOp26
Read_41/DisableCopyOnReadRead_41/DisableCopyOnRead20
Read_41/ReadVariableOpRead_41/ReadVariableOp26
Read_42/DisableCopyOnReadRead_42/DisableCopyOnRead20
Read_42/ReadVariableOpRead_42/ReadVariableOp26
Read_43/DisableCopyOnReadRead_43/DisableCopyOnRead20
Read_43/ReadVariableOpRead_43/ReadVariableOp26
Read_44/DisableCopyOnReadRead_44/DisableCopyOnRead20
Read_44/ReadVariableOpRead_44/ReadVariableOp26
Read_45/DisableCopyOnReadRead_45/DisableCopyOnRead20
Read_45/ReadVariableOpRead_45/ReadVariableOp26
Read_46/DisableCopyOnReadRead_46/DisableCopyOnRead20
Read_46/ReadVariableOpRead_46/ReadVariableOp26
Read_47/DisableCopyOnReadRead_47/DisableCopyOnRead20
Read_47/ReadVariableOpRead_47/ReadVariableOp26
Read_48/DisableCopyOnReadRead_48/DisableCopyOnRead20
Read_48/ReadVariableOpRead_48/ReadVariableOp26
Read_49/DisableCopyOnReadRead_49/DisableCopyOnRead20
Read_49/ReadVariableOpRead_49/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp26
Read_50/DisableCopyOnReadRead_50/DisableCopyOnRead20
Read_50/ReadVariableOpRead_50/ReadVariableOp26
Read_51/DisableCopyOnReadRead_51/DisableCopyOnRead20
Read_51/ReadVariableOpRead_51/ReadVariableOp26
Read_52/DisableCopyOnReadRead_52/DisableCopyOnRead20
Read_52/ReadVariableOpRead_52/ReadVariableOp26
Read_53/DisableCopyOnReadRead_53/DisableCopyOnRead20
Read_53/ReadVariableOpRead_53/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:-)
'
_user_specified_nameconv1d/kernel:+'
%
_user_specified_nameconv1d/bias:-)
'
_user_specified_nameconv3d/kernel:+'
%
_user_specified_nameconv3d/bias:95
3
_user_specified_namebatch_normalization/gamma:84
2
_user_specified_namebatch_normalization/beta:?;
9
_user_specified_name!batch_normalization/moving_mean:C?
=
_user_specified_name%#batch_normalization/moving_variance:;	7
5
_user_specified_namebatch_normalization_3/gamma::
6
4
_user_specified_namebatch_normalization_3/beta:A=
;
_user_specified_name#!batch_normalization_3/moving_mean:EA
?
_user_specified_name'%batch_normalization_3/moving_variance:/+
)
_user_specified_nameconv1d_1/kernel:-)
'
_user_specified_nameconv1d_1/bias:/+
)
_user_specified_nameconv3d_1/kernel:-)
'
_user_specified_nameconv3d_1/bias:;7
5
_user_specified_namebatch_normalization_1/gamma::6
4
_user_specified_namebatch_normalization_1/beta:A=
;
_user_specified_name#!batch_normalization_1/moving_mean:EA
?
_user_specified_name'%batch_normalization_1/moving_variance:;7
5
_user_specified_namebatch_normalization_4/gamma::6
4
_user_specified_namebatch_normalization_4/beta:A=
;
_user_specified_name#!batch_normalization_4/moving_mean:EA
?
_user_specified_name'%batch_normalization_4/moving_variance:/+
)
_user_specified_nameconv1d_2/kernel:-)
'
_user_specified_nameconv1d_2/bias:/+
)
_user_specified_nameconv3d_2/kernel:-)
'
_user_specified_nameconv3d_2/bias:;7
5
_user_specified_namebatch_normalization_2/gamma::6
4
_user_specified_namebatch_normalization_2/beta:A=
;
_user_specified_name#!batch_normalization_2/moving_mean:E A
?
_user_specified_name'%batch_normalization_2/moving_variance:;!7
5
_user_specified_namebatch_normalization_5/gamma::"6
4
_user_specified_namebatch_normalization_5/beta:A#=
;
_user_specified_name#!batch_normalization_5/moving_mean:E$A
?
_user_specified_name'%batch_normalization_5/moving_variance:,%(
&
_user_specified_namedense/kernel:*&&
$
_user_specified_name
dense/bias:.'*
(
_user_specified_namedense_1/kernel:,((
&
_user_specified_namedense_1/bias:.)*
(
_user_specified_namedense_2/kernel:,*(
&
_user_specified_namedense_2/bias:.+*
(
_user_specified_namedense_3/kernel:,,(
&
_user_specified_namedense_3/bias:.-*
(
_user_specified_namedense_4/kernel:,.(
&
_user_specified_namedense_4/bias:./*
(
_user_specified_namedense_5/kernel:,0(
&
_user_specified_namedense_5/bias:)1%
#
_user_specified_name	iteration:-2)
'
_user_specified_namelearning_rate:'3#
!
_user_specified_name	total_1:'4#
!
_user_specified_name	count_1:%5!

_user_specified_nametotal:%6!

_user_specified_namecount:=79

_output_shapes
: 

_user_specified_nameConst

Ћ
L__inference_batch_normalization_layer_call_and_return_conditional_losses_923

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identityЂbatchnorm/ReadVariableOpЂbatchnorm/ReadVariableOp_1Ђbatchnorm/ReadVariableOp_2Ђbatchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:џџџџџџџџџџџџџџџџџџ@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Ы

№
?__inference_dense_layer_call_and_return_conditional_losses_1528

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџS
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
ќ&
х
L__inference_batch_normalization_layer_call_and_return_conditional_losses_903

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityЂAssignMovingAvgЂAssignMovingAvg/ReadVariableOpЂAssignMovingAvg_1Ђ AssignMovingAvg_1/ReadVariableOpЂbatchnorm/ReadVariableOpЂbatchnorm/mul/ReadVariableOpo
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(h
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:@
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       Ђ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@Ќ
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@Д
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@Ц
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:џџџџџџџџџџџџџџџџџџ@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource


B__inference_conv1d_1_layer_call_and_return_conditional_losses_2430

inputsA
+conv1d_expanddims_1_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂ"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџd@
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :  
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@Ќ
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџd@*
paddingSAME*
strides

Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:џџџџџџџџџd@*
squeeze_dims

§џџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџd@c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџd@`
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџd@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:џџџџџџџџџd@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
б
n
R__inference_global_average_pooling3d_layer_call_and_return_conditional_losses_1312

inputs
identityk
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ: {
W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
§
]
A__inference_re_lu_5_layer_call_and_return_conditional_losses_1508

inputs
identityT
ReluReluinputs*
T0*5
_output_shapes#
!:џџџџџџџџџd@h
IdentityIdentityRelu:activations:0*
T0*5
_output_shapes#
!:џџџџџџџџџd@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:џџџџџџџџџd@:] Y
5
_output_shapes#
!:џџџџџџџџџd@
 
_user_specified_nameinputs
§
]
A__inference_re_lu_5_layer_call_and_return_conditional_losses_2816

inputs
identityT
ReluReluinputs*
T0*5
_output_shapes#
!:џџџџџџџџџd@h
IdentityIdentityRelu:activations:0*
T0*5
_output_shapes#
!:џџџџџџџџџd@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:џџџџџџџџџd@:] Y
5
_output_shapes#
!:џџџџџџџџџd@
 
_user_specified_nameinputs
е
]
A__inference_re_lu_1_layer_call_and_return_conditional_losses_2601

inputs
identityJ
ReluReluinputs*
T0*+
_output_shapes
:џџџџџџџџџd@^
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:џџџџџџџџџd@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџd@:S O
+
_output_shapes
:џџџџџџџџџd@
 
_user_specified_nameinputs
ећ
ѓ!
 __inference__traced_restore_3486
file_prefix4
assignvariableop_conv1d_kernel:t@,
assignvariableop_1_conv1d_bias:@>
 assignvariableop_2_conv3d_kernel:@,
assignvariableop_3_conv3d_bias:@:
,assignvariableop_4_batch_normalization_gamma:@9
+assignvariableop_5_batch_normalization_beta:@@
2assignvariableop_6_batch_normalization_moving_mean:@D
6assignvariableop_7_batch_normalization_moving_variance:@<
.assignvariableop_8_batch_normalization_3_gamma:@;
-assignvariableop_9_batch_normalization_3_beta:@C
5assignvariableop_10_batch_normalization_3_moving_mean:@G
9assignvariableop_11_batch_normalization_3_moving_variance:@9
#assignvariableop_12_conv1d_1_kernel:@@/
!assignvariableop_13_conv1d_1_bias:@A
#assignvariableop_14_conv3d_1_kernel:@@/
!assignvariableop_15_conv3d_1_bias:@=
/assignvariableop_16_batch_normalization_1_gamma:@<
.assignvariableop_17_batch_normalization_1_beta:@C
5assignvariableop_18_batch_normalization_1_moving_mean:@G
9assignvariableop_19_batch_normalization_1_moving_variance:@=
/assignvariableop_20_batch_normalization_4_gamma:@<
.assignvariableop_21_batch_normalization_4_beta:@C
5assignvariableop_22_batch_normalization_4_moving_mean:@G
9assignvariableop_23_batch_normalization_4_moving_variance:@9
#assignvariableop_24_conv1d_2_kernel:@@/
!assignvariableop_25_conv1d_2_bias:@A
#assignvariableop_26_conv3d_2_kernel:@@/
!assignvariableop_27_conv3d_2_bias:@=
/assignvariableop_28_batch_normalization_2_gamma:@<
.assignvariableop_29_batch_normalization_2_beta:@C
5assignvariableop_30_batch_normalization_2_moving_mean:@G
9assignvariableop_31_batch_normalization_2_moving_variance:@=
/assignvariableop_32_batch_normalization_5_gamma:@<
.assignvariableop_33_batch_normalization_5_beta:@C
5assignvariableop_34_batch_normalization_5_moving_mean:@G
9assignvariableop_35_batch_normalization_5_moving_variance:@2
 assignvariableop_36_dense_kernel:@,
assignvariableop_37_dense_bias:4
"assignvariableop_38_dense_1_kernel:@.
 assignvariableop_39_dense_1_bias:4
"assignvariableop_40_dense_2_kernel:
 .
 assignvariableop_41_dense_2_bias: 4
"assignvariableop_42_dense_3_kernel: .
 assignvariableop_43_dense_3_bias:4
"assignvariableop_44_dense_4_kernel:.
 assignvariableop_45_dense_4_bias:4
"assignvariableop_46_dense_5_kernel:.
 assignvariableop_47_dense_5_bias:'
assignvariableop_48_iteration:	 +
!assignvariableop_49_learning_rate: %
assignvariableop_50_total_1: %
assignvariableop_51_count_1: #
assignvariableop_52_total: #
assignvariableop_53_count: 
identity_55ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_28ЂAssignVariableOp_29ЂAssignVariableOp_3ЂAssignVariableOp_30ЂAssignVariableOp_31ЂAssignVariableOp_32ЂAssignVariableOp_33ЂAssignVariableOp_34ЂAssignVariableOp_35ЂAssignVariableOp_36ЂAssignVariableOp_37ЂAssignVariableOp_38ЂAssignVariableOp_39ЂAssignVariableOp_4ЂAssignVariableOp_40ЂAssignVariableOp_41ЂAssignVariableOp_42ЂAssignVariableOp_43ЂAssignVariableOp_44ЂAssignVariableOp_45ЂAssignVariableOp_46ЂAssignVariableOp_47ЂAssignVariableOp_48ЂAssignVariableOp_49ЂAssignVariableOp_5ЂAssignVariableOp_50ЂAssignVariableOp_51ЂAssignVariableOp_52ЂAssignVariableOp_53ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*А
valueІBЃ7B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHп
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*
valuexBv7B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Д
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*ђ
_output_shapesп
м:::::::::::::::::::::::::::::::::::::::::::::::::::::::*E
dtypes;
927	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:Б
AssignVariableOpAssignVariableOpassignvariableop_conv1d_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:Е
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv1d_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:З
AssignVariableOp_2AssignVariableOp assignvariableop_2_conv3d_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Е
AssignVariableOp_3AssignVariableOpassignvariableop_3_conv3d_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_4AssignVariableOp,assignvariableop_4_batch_normalization_gammaIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_5AssignVariableOp+assignvariableop_5_batch_normalization_betaIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_6AssignVariableOp2assignvariableop_6_batch_normalization_moving_meanIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_7AssignVariableOp6assignvariableop_7_batch_normalization_moving_varianceIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_8AssignVariableOp.assignvariableop_8_batch_normalization_3_gammaIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_9AssignVariableOp-assignvariableop_9_batch_normalization_3_betaIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_10AssignVariableOp5assignvariableop_10_batch_normalization_3_moving_meanIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:в
AssignVariableOp_11AssignVariableOp9assignvariableop_11_batch_normalization_3_moving_varianceIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_12AssignVariableOp#assignvariableop_12_conv1d_1_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_13AssignVariableOp!assignvariableop_13_conv1d_1_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_14AssignVariableOp#assignvariableop_14_conv3d_1_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_15AssignVariableOp!assignvariableop_15_conv3d_1_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_16AssignVariableOp/assignvariableop_16_batch_normalization_1_gammaIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_17AssignVariableOp.assignvariableop_17_batch_normalization_1_betaIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_18AssignVariableOp5assignvariableop_18_batch_normalization_1_moving_meanIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:в
AssignVariableOp_19AssignVariableOp9assignvariableop_19_batch_normalization_1_moving_varianceIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_20AssignVariableOp/assignvariableop_20_batch_normalization_4_gammaIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_21AssignVariableOp.assignvariableop_21_batch_normalization_4_betaIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_22AssignVariableOp5assignvariableop_22_batch_normalization_4_moving_meanIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:в
AssignVariableOp_23AssignVariableOp9assignvariableop_23_batch_normalization_4_moving_varianceIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_24AssignVariableOp#assignvariableop_24_conv1d_2_kernelIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_25AssignVariableOp!assignvariableop_25_conv1d_2_biasIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_26AssignVariableOp#assignvariableop_26_conv3d_2_kernelIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_27AssignVariableOp!assignvariableop_27_conv3d_2_biasIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_28AssignVariableOp/assignvariableop_28_batch_normalization_2_gammaIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_29AssignVariableOp.assignvariableop_29_batch_normalization_2_betaIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_30AssignVariableOp5assignvariableop_30_batch_normalization_2_moving_meanIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:в
AssignVariableOp_31AssignVariableOp9assignvariableop_31_batch_normalization_2_moving_varianceIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_32AssignVariableOp/assignvariableop_32_batch_normalization_5_gammaIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_33AssignVariableOp.assignvariableop_33_batch_normalization_5_betaIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_34AssignVariableOp5assignvariableop_34_batch_normalization_5_moving_meanIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:в
AssignVariableOp_35AssignVariableOp9assignvariableop_35_batch_normalization_5_moving_varianceIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:Й
AssignVariableOp_36AssignVariableOp assignvariableop_36_dense_kernelIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:З
AssignVariableOp_37AssignVariableOpassignvariableop_37_dense_biasIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOp_38AssignVariableOp"assignvariableop_38_dense_1_kernelIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:Й
AssignVariableOp_39AssignVariableOp assignvariableop_39_dense_1_biasIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOp_40AssignVariableOp"assignvariableop_40_dense_2_kernelIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:Й
AssignVariableOp_41AssignVariableOp assignvariableop_41_dense_2_biasIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOp_42AssignVariableOp"assignvariableop_42_dense_3_kernelIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:Й
AssignVariableOp_43AssignVariableOp assignvariableop_43_dense_3_biasIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOp_44AssignVariableOp"assignvariableop_44_dense_4_kernelIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:Й
AssignVariableOp_45AssignVariableOp assignvariableop_45_dense_4_biasIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOp_46AssignVariableOp"assignvariableop_46_dense_5_kernelIdentity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:Й
AssignVariableOp_47AssignVariableOp assignvariableop_47_dense_5_biasIdentity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0	*
_output_shapes
:Ж
AssignVariableOp_48AssignVariableOpassignvariableop_48_iterationIdentity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_49AssignVariableOp!assignvariableop_49_learning_rateIdentity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_50AssignVariableOpassignvariableop_50_total_1Identity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_51AssignVariableOpassignvariableop_51_count_1Identity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:В
AssignVariableOp_52AssignVariableOpassignvariableop_52_totalIdentity_52:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:В
AssignVariableOp_53AssignVariableOpassignvariableop_53_countIdentity_53:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 ѓ	
Identity_54Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_55IdentityIdentity_54:output:0^NoOp_1*
T0*
_output_shapes
: М	
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_55Identity_55:output:0*(
_construction_contextkEagerRuntime*
_input_shapesp
n: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:-)
'
_user_specified_nameconv1d/kernel:+'
%
_user_specified_nameconv1d/bias:-)
'
_user_specified_nameconv3d/kernel:+'
%
_user_specified_nameconv3d/bias:95
3
_user_specified_namebatch_normalization/gamma:84
2
_user_specified_namebatch_normalization/beta:?;
9
_user_specified_name!batch_normalization/moving_mean:C?
=
_user_specified_name%#batch_normalization/moving_variance:;	7
5
_user_specified_namebatch_normalization_3/gamma::
6
4
_user_specified_namebatch_normalization_3/beta:A=
;
_user_specified_name#!batch_normalization_3/moving_mean:EA
?
_user_specified_name'%batch_normalization_3/moving_variance:/+
)
_user_specified_nameconv1d_1/kernel:-)
'
_user_specified_nameconv1d_1/bias:/+
)
_user_specified_nameconv3d_1/kernel:-)
'
_user_specified_nameconv3d_1/bias:;7
5
_user_specified_namebatch_normalization_1/gamma::6
4
_user_specified_namebatch_normalization_1/beta:A=
;
_user_specified_name#!batch_normalization_1/moving_mean:EA
?
_user_specified_name'%batch_normalization_1/moving_variance:;7
5
_user_specified_namebatch_normalization_4/gamma::6
4
_user_specified_namebatch_normalization_4/beta:A=
;
_user_specified_name#!batch_normalization_4/moving_mean:EA
?
_user_specified_name'%batch_normalization_4/moving_variance:/+
)
_user_specified_nameconv1d_2/kernel:-)
'
_user_specified_nameconv1d_2/bias:/+
)
_user_specified_nameconv3d_2/kernel:-)
'
_user_specified_nameconv3d_2/bias:;7
5
_user_specified_namebatch_normalization_2/gamma::6
4
_user_specified_namebatch_normalization_2/beta:A=
;
_user_specified_name#!batch_normalization_2/moving_mean:E A
?
_user_specified_name'%batch_normalization_2/moving_variance:;!7
5
_user_specified_namebatch_normalization_5/gamma::"6
4
_user_specified_namebatch_normalization_5/beta:A#=
;
_user_specified_name#!batch_normalization_5/moving_mean:E$A
?
_user_specified_name'%batch_normalization_5/moving_variance:,%(
&
_user_specified_namedense/kernel:*&&
$
_user_specified_name
dense/bias:.'*
(
_user_specified_namedense_1/kernel:,((
&
_user_specified_namedense_1/bias:.)*
(
_user_specified_namedense_2/kernel:,*(
&
_user_specified_namedense_2/bias:.+*
(
_user_specified_namedense_3/kernel:,,(
&
_user_specified_namedense_3/bias:.-*
(
_user_specified_namedense_4/kernel:,.(
&
_user_specified_namedense_4/bias:./*
(
_user_specified_namedense_5/kernel:,0(
&
_user_specified_namedense_5/bias:)1%
#
_user_specified_name	iteration:-2)
'
_user_specified_namelearning_rate:'3#
!
_user_specified_name	total_1:'4#
!
_user_specified_name	count_1:%5!

_user_specified_nametotal:%6!

_user_specified_namecount
д
Н
N__inference_batch_normalization_3_layer_call_and_return_conditional_losses_967

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0ћ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*j
_output_shapesX
V:8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@:@:@:@:@:*
data_formatNDHWC*
epsilon%o:*
exponential_avg_factor%
з#<Ц
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(а
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*N
_output_shapes<
::8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@А
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:v r
N
_output_shapes<
::8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
б	
Я
4__inference_batch_normalization_2_layer_call_fn_2680

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1207|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:џџџџџџџџџџџџџџџџџџ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs:$ 

_user_specified_name2670:$ 

_user_specified_name2672:$ 

_user_specified_name2674:$ 

_user_specified_name2676
Я	
Я
4__inference_batch_normalization_1_layer_call_fn_2462

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1045|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:џџџџџџџџџџџџџџџџџџ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs:$ 

_user_specified_name2452:$ 

_user_specified_name2454:$ 

_user_specified_name2456:$ 

_user_specified_name2458
ќ

§
@__inference_conv3d_layer_call_and_return_conditional_losses_1330

inputs<
conv3d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:@*
dtype0 
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:џџџџџџџџџd@*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:џџџџџџџџџd@m
IdentityIdentityBiasAdd:output:0^NoOp*
T0*5
_output_shapes#
!:џџџџџџџџџd@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:џџџџџџџџџd: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:] Y
5
_output_shapes#
!:џџџџџџџџџd
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Џ
 
'__inference_conv3d_2_layer_call_fn_2644

inputs%
unknown:@@
	unknown_0:@
identityЂStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:џџџџџџџџџd@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *K
fFRD
B__inference_conv3d_2_layer_call_and_return_conditional_losses_1460}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:џџџџџџџџџd@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:џџџџџџџџџd@: : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:џџџџџџџџџd@
 
_user_specified_nameinputs:$ 

_user_specified_name2638:$ 

_user_specified_name2640


O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_2386

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0э
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*j
_output_shapesX
V:8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@:@:@:@:@:*
data_formatNDHWC*
epsilon%o:*
is_training( 
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*N
_output_shapes<
::8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:v r
N
_output_shapes<
::8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
є	
ђ
A__inference_dense_2_layer_call_and_return_conditional_losses_1567

inputs0
matmul_readvariableop_resource:
 -
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
 *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ

 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Ъ	
Э
2__inference_batch_normalization_layer_call_fn_2257

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *U
fPRN
L__inference_batch_normalization_layer_call_and_return_conditional_losses_903|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:џџџџџџџџџџџџџџџџџџ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs:$ 

_user_specified_name2247:$ 

_user_specified_name2249:$ 

_user_specified_name2251:$ 

_user_specified_name2253"эL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*љ
serving_defaultх
?
input_14
serving_default_input_1:0џџџџџџџџџdt
I
input_2>
serving_default_input_2:0џџџџџџџџџd;
dense_50
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:яв
Ы
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer_with_weights-6
layer-10
layer_with_weights-7
layer-11
layer-12
layer-13
layer_with_weights-8
layer-14
layer_with_weights-9
layer-15
layer_with_weights-10
layer-16
layer_with_weights-11
layer-17
layer-18
layer-19
layer-20
layer-21
layer_with_weights-12
layer-22
layer_with_weights-13
layer-23
layer-24
layer_with_weights-14
layer-25
layer_with_weights-15
layer-26
layer_with_weights-16
layer-27
layer_with_weights-17
layer-28
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses
$_default_save_signature
%	optimizer
&
signatures"
_tf_keras_network
"
_tf_keras_input_layer
"
_tf_keras_input_layer
н
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses

-kernel
.bias
 /_jit_compiled_convolution_op"
_tf_keras_layer
н
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses

6kernel
7bias
 8_jit_compiled_convolution_op"
_tf_keras_layer
ъ
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses
?axis
	@gamma
Abeta
Bmoving_mean
Cmoving_variance"
_tf_keras_layer
ъ
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
H__call__
*I&call_and_return_all_conditional_losses
Jaxis
	Kgamma
Lbeta
Mmoving_mean
Nmoving_variance"
_tf_keras_layer
Ѕ
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses"
_tf_keras_layer
Ѕ
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses"
_tf_keras_layer
н
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses

akernel
bbias
 c_jit_compiled_convolution_op"
_tf_keras_layer
н
d	variables
etrainable_variables
fregularization_losses
g	keras_api
h__call__
*i&call_and_return_all_conditional_losses

jkernel
kbias
 l_jit_compiled_convolution_op"
_tf_keras_layer
ъ
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
q__call__
*r&call_and_return_all_conditional_losses
saxis
	tgamma
ubeta
vmoving_mean
wmoving_variance"
_tf_keras_layer
э
x	variables
ytrainable_variables
zregularization_losses
{	keras_api
|__call__
*}&call_and_return_all_conditional_losses
~axis
	gamma
	beta
moving_mean
moving_variance"
_tf_keras_layer
Ћ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ћ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
ц
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias
!_jit_compiled_convolution_op"
_tf_keras_layer
ц
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias
! _jit_compiled_convolution_op"
_tf_keras_layer
ѕ
Ё	variables
Ђtrainable_variables
Ѓregularization_losses
Є	keras_api
Ѕ__call__
+І&call_and_return_all_conditional_losses
	Їaxis

Јgamma
	Љbeta
Њmoving_mean
Ћmoving_variance"
_tf_keras_layer
ѕ
Ќ	variables
­trainable_variables
Ўregularization_losses
Џ	keras_api
А__call__
+Б&call_and_return_all_conditional_losses
	Вaxis

Гgamma
	Дbeta
Еmoving_mean
Жmoving_variance"
_tf_keras_layer
Ћ
З	variables
Иtrainable_variables
Йregularization_losses
К	keras_api
Л__call__
+М&call_and_return_all_conditional_losses"
_tf_keras_layer
Ћ
Н	variables
Оtrainable_variables
Пregularization_losses
Р	keras_api
С__call__
+Т&call_and_return_all_conditional_losses"
_tf_keras_layer
Ћ
У	variables
Фtrainable_variables
Хregularization_losses
Ц	keras_api
Ч__call__
+Ш&call_and_return_all_conditional_losses"
_tf_keras_layer
Ћ
Щ	variables
Ъtrainable_variables
Ыregularization_losses
Ь	keras_api
Э__call__
+Ю&call_and_return_all_conditional_losses"
_tf_keras_layer
У
Я	variables
аtrainable_variables
бregularization_losses
в	keras_api
г__call__
+д&call_and_return_all_conditional_losses
еkernel
	жbias"
_tf_keras_layer
У
з	variables
иtrainable_variables
йregularization_losses
к	keras_api
л__call__
+м&call_and_return_all_conditional_losses
нkernel
	оbias"
_tf_keras_layer
Ћ
п	variables
рtrainable_variables
сregularization_losses
т	keras_api
у__call__
+ф&call_and_return_all_conditional_losses"
_tf_keras_layer
У
х	variables
цtrainable_variables
чregularization_losses
ш	keras_api
щ__call__
+ъ&call_and_return_all_conditional_losses
ыkernel
	ьbias"
_tf_keras_layer
У
э	variables
юtrainable_variables
яregularization_losses
№	keras_api
ё__call__
+ђ&call_and_return_all_conditional_losses
ѓkernel
	єbias"
_tf_keras_layer
У
ѕ	variables
іtrainable_variables
їregularization_losses
ј	keras_api
љ__call__
+њ&call_and_return_all_conditional_losses
ћkernel
	ќbias"
_tf_keras_layer
У
§	variables
ўtrainable_variables
џregularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias"
_tf_keras_layer
Б
-0
.1
62
73
@4
A5
B6
C7
K8
L9
M10
N11
a12
b13
j14
k15
t16
u17
v18
w19
20
21
22
23
24
25
26
27
Ј28
Љ29
Њ30
Ћ31
Г32
Д33
Е34
Ж35
е36
ж37
н38
о39
ы40
ь41
ѓ42
є43
ћ44
ќ45
46
47"
trackable_list_wrapper
Ы
-0
.1
62
73
@4
A5
K6
L7
a8
b9
j10
k11
t12
u13
14
15
16
17
18
19
Ј20
Љ21
Г22
Д23
е24
ж25
н26
о27
ы28
ь29
ѓ30
є31
ћ32
ќ33
34
35"
trackable_list_wrapper
 "
trackable_list_wrapper
Я
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
$_default_save_signature
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object
Щ
trace_0
trace_12
)__inference_Full_Model_layer_call_fn_1850
)__inference_Full_Model_layer_call_fn_1952Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0ztrace_1
џ
trace_0
trace_12Ф
D__inference_Full_Model_layer_call_and_return_conditional_losses_1620
D__inference_Full_Model_layer_call_and_return_conditional_losses_1748Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0ztrace_1
вBЯ
__inference__wrapped_model_869input_1input_2"
В
FullArgSpec
args

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
n

_variables
_iterations
_learning_rate
_update_step_xla"
experimentalOptimizer
-
serving_default"
signature_map
.
-0
.1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses"
_generic_user_object
с
trace_02Т
%__inference_conv1d_layer_call_fn_2210
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
ќ
trace_02н
@__inference_conv1d_layer_call_and_return_conditional_losses_2225
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
#:!t@2conv1d/kernel
:@2conv1d/bias
Њ2ЇЄ
В
FullArgSpec
args
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
.
60
71"
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
с
trace_02Т
%__inference_conv3d_layer_call_fn_2234
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
ќ
 trace_02н
@__inference_conv3d_layer_call_and_return_conditional_losses_2244
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z trace_0
+:)@2conv3d/kernel
:@2conv3d/bias
Њ2ЇЄ
В
FullArgSpec
args
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
<
@0
A1
B2
C3"
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
Ёnon_trainable_variables
Ђlayers
Ѓmetrics
 Єlayer_regularization_losses
Ѕlayer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
л
Іtrace_0
Їtrace_12 
2__inference_batch_normalization_layer_call_fn_2257
2__inference_batch_normalization_layer_call_fn_2270Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zІtrace_0zЇtrace_1

Јtrace_0
Љtrace_12ж
M__inference_batch_normalization_layer_call_and_return_conditional_losses_2304
M__inference_batch_normalization_layer_call_and_return_conditional_losses_2324Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЈtrace_0zЉtrace_1
 "
trackable_list_wrapper
':%@2batch_normalization/gamma
&:$@2batch_normalization/beta
/:-@ (2batch_normalization/moving_mean
3:1@ (2#batch_normalization/moving_variance
<
K0
L1
M2
N3"
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
Њnon_trainable_variables
Ћlayers
Ќmetrics
 ­layer_regularization_losses
Ўlayer_metrics
D	variables
Etrainable_variables
Fregularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses"
_generic_user_object
п
Џtrace_0
Аtrace_12Є
4__inference_batch_normalization_3_layer_call_fn_2337
4__inference_batch_normalization_3_layer_call_fn_2350Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЏtrace_0zАtrace_1

Бtrace_0
Вtrace_12к
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_2368
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_2386Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zБtrace_0zВtrace_1
 "
trackable_list_wrapper
):'@2batch_normalization_3/gamma
(:&@2batch_normalization_3/beta
1:/@ (2!batch_normalization_3/moving_mean
5:3@ (2%batch_normalization_3/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
Гnon_trainable_variables
Дlayers
Еmetrics
 Жlayer_regularization_losses
Зlayer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses"
_generic_user_object
р
Иtrace_02С
$__inference_re_lu_layer_call_fn_2391
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zИtrace_0
ћ
Йtrace_02м
?__inference_re_lu_layer_call_and_return_conditional_losses_2396
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЙtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
Кnon_trainable_variables
Лlayers
Мmetrics
 Нlayer_regularization_losses
Оlayer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
т
Пtrace_02У
&__inference_re_lu_3_layer_call_fn_2401
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zПtrace_0
§
Рtrace_02о
A__inference_re_lu_3_layer_call_and_return_conditional_losses_2406
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zРtrace_0
.
a0
b1"
trackable_list_wrapper
.
a0
b1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
Сnon_trainable_variables
Тlayers
Уmetrics
 Фlayer_regularization_losses
Хlayer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
у
Цtrace_02Ф
'__inference_conv1d_1_layer_call_fn_2415
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЦtrace_0
ў
Чtrace_02п
B__inference_conv1d_1_layer_call_and_return_conditional_losses_2430
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЧtrace_0
%:#@@2conv1d_1/kernel
:@2conv1d_1/bias
Њ2ЇЄ
В
FullArgSpec
args
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
.
j0
k1"
trackable_list_wrapper
.
j0
k1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
Шnon_trainable_variables
Щlayers
Ъmetrics
 Ыlayer_regularization_losses
Ьlayer_metrics
d	variables
etrainable_variables
fregularization_losses
h__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses"
_generic_user_object
у
Эtrace_02Ф
'__inference_conv3d_1_layer_call_fn_2439
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЭtrace_0
ў
Юtrace_02п
B__inference_conv3d_1_layer_call_and_return_conditional_losses_2449
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЮtrace_0
-:+@@2conv3d_1/kernel
:@2conv3d_1/bias
Њ2ЇЄ
В
FullArgSpec
args
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
<
t0
u1
v2
w3"
trackable_list_wrapper
.
t0
u1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
Яnon_trainable_variables
аlayers
бmetrics
 вlayer_regularization_losses
гlayer_metrics
m	variables
ntrainable_variables
oregularization_losses
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
п
дtrace_0
еtrace_12Є
4__inference_batch_normalization_1_layer_call_fn_2462
4__inference_batch_normalization_1_layer_call_fn_2475Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zдtrace_0zеtrace_1

жtrace_0
зtrace_12к
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2509
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2529Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zжtrace_0zзtrace_1
 "
trackable_list_wrapper
):'@2batch_normalization_1/gamma
(:&@2batch_normalization_1/beta
1:/@ (2!batch_normalization_1/moving_mean
5:3@ (2%batch_normalization_1/moving_variance
?
0
1
2
3"
trackable_list_wrapper
/
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
иnon_trainable_variables
йlayers
кmetrics
 лlayer_regularization_losses
мlayer_metrics
x	variables
ytrainable_variables
zregularization_losses
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
п
нtrace_0
оtrace_12Є
4__inference_batch_normalization_4_layer_call_fn_2542
4__inference_batch_normalization_4_layer_call_fn_2555Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zнtrace_0zоtrace_1

пtrace_0
рtrace_12к
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_2573
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_2591Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zпtrace_0zрtrace_1
 "
trackable_list_wrapper
):'@2batch_normalization_4/gamma
(:&@2batch_normalization_4/beta
1:/@ (2!batch_normalization_4/moving_mean
5:3@ (2%batch_normalization_4/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
сnon_trainable_variables
тlayers
уmetrics
 фlayer_regularization_losses
хlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
т
цtrace_02У
&__inference_re_lu_1_layer_call_fn_2596
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zцtrace_0
§
чtrace_02о
A__inference_re_lu_1_layer_call_and_return_conditional_losses_2601
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zчtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
шnon_trainable_variables
щlayers
ъmetrics
 ыlayer_regularization_losses
ьlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
т
эtrace_02У
&__inference_re_lu_4_layer_call_fn_2606
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zэtrace_0
§
юtrace_02о
A__inference_re_lu_4_layer_call_and_return_conditional_losses_2611
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zюtrace_0
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
яnon_trainable_variables
№layers
ёmetrics
 ђlayer_regularization_losses
ѓlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
у
єtrace_02Ф
'__inference_conv1d_2_layer_call_fn_2620
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zєtrace_0
ў
ѕtrace_02п
B__inference_conv1d_2_layer_call_and_return_conditional_losses_2635
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zѕtrace_0
%:#@@2conv1d_2/kernel
:@2conv1d_2/bias
Њ2ЇЄ
В
FullArgSpec
args
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
іnon_trainable_variables
їlayers
јmetrics
 љlayer_regularization_losses
њlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
у
ћtrace_02Ф
'__inference_conv3d_2_layer_call_fn_2644
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zћtrace_0
ў
ќtrace_02п
B__inference_conv3d_2_layer_call_and_return_conditional_losses_2654
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zќtrace_0
-:+@@2conv3d_2/kernel
:@2conv3d_2/bias
Њ2ЇЄ
В
FullArgSpec
args
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
@
Ј0
Љ1
Њ2
Ћ3"
trackable_list_wrapper
0
Ј0
Љ1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
§non_trainable_variables
ўlayers
џmetrics
 layer_regularization_losses
layer_metrics
Ё	variables
Ђtrainable_variables
Ѓregularization_losses
Ѕ__call__
+І&call_and_return_all_conditional_losses
'І"call_and_return_conditional_losses"
_generic_user_object
п
trace_0
trace_12Є
4__inference_batch_normalization_2_layer_call_fn_2667
4__inference_batch_normalization_2_layer_call_fn_2680Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0ztrace_1

trace_0
trace_12к
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2714
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2734Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0ztrace_1
 "
trackable_list_wrapper
):'@2batch_normalization_2/gamma
(:&@2batch_normalization_2/beta
1:/@ (2!batch_normalization_2/moving_mean
5:3@ (2%batch_normalization_2/moving_variance
@
Г0
Д1
Е2
Ж3"
trackable_list_wrapper
0
Г0
Д1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ќ	variables
­trainable_variables
Ўregularization_losses
А__call__
+Б&call_and_return_all_conditional_losses
'Б"call_and_return_conditional_losses"
_generic_user_object
п
trace_0
trace_12Є
4__inference_batch_normalization_5_layer_call_fn_2747
4__inference_batch_normalization_5_layer_call_fn_2760Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0ztrace_1

trace_0
trace_12к
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_2778
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_2796Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0ztrace_1
 "
trackable_list_wrapper
):'@2batch_normalization_5/gamma
(:&@2batch_normalization_5/beta
1:/@ (2!batch_normalization_5/moving_mean
5:3@ (2%batch_normalization_5/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
З	variables
Иtrainable_variables
Йregularization_losses
Л__call__
+М&call_and_return_all_conditional_losses
'М"call_and_return_conditional_losses"
_generic_user_object
т
trace_02У
&__inference_re_lu_2_layer_call_fn_2801
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
§
trace_02о
A__inference_re_lu_2_layer_call_and_return_conditional_losses_2806
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Н	variables
Оtrainable_variables
Пregularization_losses
С__call__
+Т&call_and_return_all_conditional_losses
'Т"call_and_return_conditional_losses"
_generic_user_object
т
trace_02У
&__inference_re_lu_5_layer_call_fn_2811
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
§
trace_02о
A__inference_re_lu_5_layer_call_and_return_conditional_losses_2816
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
non_trainable_variables
layers
metrics
  layer_regularization_losses
Ёlayer_metrics
У	variables
Фtrainable_variables
Хregularization_losses
Ч__call__
+Ш&call_and_return_all_conditional_losses
'Ш"call_and_return_conditional_losses"
_generic_user_object

Ђtrace_02с
7__inference_global_average_pooling1d_layer_call_fn_2821Ѕ
В
FullArgSpec
args
jinputs
jmask
varargs
 
varkw
 
defaultsЂ

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЂtrace_0

Ѓtrace_02ќ
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_2827Ѕ
В
FullArgSpec
args
jinputs
jmask
varargs
 
varkw
 
defaultsЂ

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЃtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Єnon_trainable_variables
Ѕlayers
Іmetrics
 Їlayer_regularization_losses
Јlayer_metrics
Щ	variables
Ъtrainable_variables
Ыregularization_losses
Э__call__
+Ю&call_and_return_all_conditional_losses
'Ю"call_and_return_conditional_losses"
_generic_user_object
ѓ
Љtrace_02д
7__inference_global_average_pooling3d_layer_call_fn_2832
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЉtrace_0

Њtrace_02я
R__inference_global_average_pooling3d_layer_call_and_return_conditional_losses_2838
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЊtrace_0
0
е0
ж1"
trackable_list_wrapper
0
е0
ж1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Ћnon_trainable_variables
Ќlayers
­metrics
 Ўlayer_regularization_losses
Џlayer_metrics
Я	variables
аtrainable_variables
бregularization_losses
г__call__
+д&call_and_return_all_conditional_losses
'д"call_and_return_conditional_losses"
_generic_user_object
р
Аtrace_02С
$__inference_dense_layer_call_fn_2847
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zАtrace_0
ћ
Бtrace_02м
?__inference_dense_layer_call_and_return_conditional_losses_2858
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zБtrace_0
:@2dense/kernel
:2
dense/bias
0
н0
о1"
trackable_list_wrapper
0
н0
о1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Вnon_trainable_variables
Гlayers
Дmetrics
 Еlayer_regularization_losses
Жlayer_metrics
з	variables
иtrainable_variables
йregularization_losses
л__call__
+м&call_and_return_all_conditional_losses
'м"call_and_return_conditional_losses"
_generic_user_object
т
Зtrace_02У
&__inference_dense_1_layer_call_fn_2867
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЗtrace_0
§
Иtrace_02о
A__inference_dense_1_layer_call_and_return_conditional_losses_2878
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zИtrace_0
 :@2dense_1/kernel
:2dense_1/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Йnon_trainable_variables
Кlayers
Лmetrics
 Мlayer_regularization_losses
Нlayer_metrics
п	variables
рtrainable_variables
сregularization_losses
у__call__
+ф&call_and_return_all_conditional_losses
'ф"call_and_return_conditional_losses"
_generic_user_object
ц
Оtrace_02Ч
*__inference_concatenate_layer_call_fn_2884
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zОtrace_0

Пtrace_02т
E__inference_concatenate_layer_call_and_return_conditional_losses_2891
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zПtrace_0
0
ы0
ь1"
trackable_list_wrapper
0
ы0
ь1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Рnon_trainable_variables
Сlayers
Тmetrics
 Уlayer_regularization_losses
Фlayer_metrics
х	variables
цtrainable_variables
чregularization_losses
щ__call__
+ъ&call_and_return_all_conditional_losses
'ъ"call_and_return_conditional_losses"
_generic_user_object
т
Хtrace_02У
&__inference_dense_2_layer_call_fn_2900
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zХtrace_0
§
Цtrace_02о
A__inference_dense_2_layer_call_and_return_conditional_losses_2910
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЦtrace_0
 :
 2dense_2/kernel
: 2dense_2/bias
0
ѓ0
є1"
trackable_list_wrapper
0
ѓ0
є1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Чnon_trainable_variables
Шlayers
Щmetrics
 Ъlayer_regularization_losses
Ыlayer_metrics
э	variables
юtrainable_variables
яregularization_losses
ё__call__
+ђ&call_and_return_all_conditional_losses
'ђ"call_and_return_conditional_losses"
_generic_user_object
т
Ьtrace_02У
&__inference_dense_3_layer_call_fn_2919
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЬtrace_0
§
Эtrace_02о
A__inference_dense_3_layer_call_and_return_conditional_losses_2929
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЭtrace_0
 : 2dense_3/kernel
:2dense_3/bias
0
ћ0
ќ1"
trackable_list_wrapper
0
ћ0
ќ1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Юnon_trainable_variables
Яlayers
аmetrics
 бlayer_regularization_losses
вlayer_metrics
ѕ	variables
іtrainable_variables
їregularization_losses
љ__call__
+њ&call_and_return_all_conditional_losses
'њ"call_and_return_conditional_losses"
_generic_user_object
т
гtrace_02У
&__inference_dense_4_layer_call_fn_2938
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zгtrace_0
§
дtrace_02о
A__inference_dense_4_layer_call_and_return_conditional_losses_2948
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zдtrace_0
 :2dense_4/kernel
:2dense_4/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
еnon_trainable_variables
жlayers
зmetrics
 иlayer_regularization_losses
йlayer_metrics
§	variables
ўtrainable_variables
џregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
т
кtrace_02У
&__inference_dense_5_layer_call_fn_2957
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zкtrace_0
§
лtrace_02о
A__inference_dense_5_layer_call_and_return_conditional_losses_2968
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zлtrace_0
 :2dense_5/kernel
:2dense_5/bias
|
B0
C1
M2
N3
v4
w5
6
7
Њ8
Ћ9
Е10
Ж11"
trackable_list_wrapper
ў
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28"
trackable_list_wrapper
0
м0
н1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ёBю
)__inference_Full_Model_layer_call_fn_1850input_1input_2"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ёBю
)__inference_Full_Model_layer_call_fn_1952input_1input_2"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
D__inference_Full_Model_layer_call_and_return_conditional_losses_1620input_1input_2"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
D__inference_Full_Model_layer_call_and_return_conditional_losses_1748input_1input_2"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
(
0"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
Е2ВЏ
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
рBн
"__inference_signature_wrapper_2201input_1input_2"Є
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 '

kwonlyargs
	jinput_1
	jinput_2
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЯBЬ
%__inference_conv1d_layer_call_fn_2210inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ъBч
@__inference_conv1d_layer_call_and_return_conditional_losses_2225inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЯBЬ
%__inference_conv3d_layer_call_fn_2234inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ъBч
@__inference_conv3d_layer_call_and_return_conditional_losses_2244inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
.
B0
C1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
№Bэ
2__inference_batch_normalization_layer_call_fn_2257inputs"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
№Bэ
2__inference_batch_normalization_layer_call_fn_2270inputs"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
M__inference_batch_normalization_layer_call_and_return_conditional_losses_2304inputs"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
M__inference_batch_normalization_layer_call_and_return_conditional_losses_2324inputs"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
.
M0
N1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ђBя
4__inference_batch_normalization_3_layer_call_fn_2337inputs"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ђBя
4__inference_batch_normalization_3_layer_call_fn_2350inputs"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_2368inputs"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_2386inputs"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЮBЫ
$__inference_re_lu_layer_call_fn_2391inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
щBц
?__inference_re_lu_layer_call_and_return_conditional_losses_2396inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
аBЭ
&__inference_re_lu_3_layer_call_fn_2401inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ыBш
A__inference_re_lu_3_layer_call_and_return_conditional_losses_2406inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
бBЮ
'__inference_conv1d_1_layer_call_fn_2415inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ьBщ
B__inference_conv1d_1_layer_call_and_return_conditional_losses_2430inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
бBЮ
'__inference_conv3d_1_layer_call_fn_2439inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ьBщ
B__inference_conv3d_1_layer_call_and_return_conditional_losses_2449inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
.
v0
w1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ђBя
4__inference_batch_normalization_1_layer_call_fn_2462inputs"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ђBя
4__inference_batch_normalization_1_layer_call_fn_2475inputs"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2509inputs"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2529inputs"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ђBя
4__inference_batch_normalization_4_layer_call_fn_2542inputs"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ђBя
4__inference_batch_normalization_4_layer_call_fn_2555inputs"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_2573inputs"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_2591inputs"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
аBЭ
&__inference_re_lu_1_layer_call_fn_2596inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ыBш
A__inference_re_lu_1_layer_call_and_return_conditional_losses_2601inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
аBЭ
&__inference_re_lu_4_layer_call_fn_2606inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ыBш
A__inference_re_lu_4_layer_call_and_return_conditional_losses_2611inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
бBЮ
'__inference_conv1d_2_layer_call_fn_2620inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ьBщ
B__inference_conv1d_2_layer_call_and_return_conditional_losses_2635inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
бBЮ
'__inference_conv3d_2_layer_call_fn_2644inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ьBщ
B__inference_conv3d_2_layer_call_and_return_conditional_losses_2654inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
0
Њ0
Ћ1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ђBя
4__inference_batch_normalization_2_layer_call_fn_2667inputs"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ђBя
4__inference_batch_normalization_2_layer_call_fn_2680inputs"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2714inputs"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2734inputs"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
0
Е0
Ж1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ђBя
4__inference_batch_normalization_5_layer_call_fn_2747inputs"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ђBя
4__inference_batch_normalization_5_layer_call_fn_2760inputs"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_2778inputs"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_2796inputs"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
аBЭ
&__inference_re_lu_2_layer_call_fn_2801inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ыBш
A__inference_re_lu_2_layer_call_and_return_conditional_losses_2806inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
аBЭ
&__inference_re_lu_5_layer_call_fn_2811inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ыBш
A__inference_re_lu_5_layer_call_and_return_conditional_losses_2816inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
щBц
7__inference_global_average_pooling1d_layer_call_fn_2821inputs" 
В
FullArgSpec
args
jinputs
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_2827inputs" 
В
FullArgSpec
args
jinputs
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
сBо
7__inference_global_average_pooling3d_layer_call_fn_2832inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ќBљ
R__inference_global_average_pooling3d_layer_call_and_return_conditional_losses_2838inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЮBЫ
$__inference_dense_layer_call_fn_2847inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
щBц
?__inference_dense_layer_call_and_return_conditional_losses_2858inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
аBЭ
&__inference_dense_1_layer_call_fn_2867inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ыBш
A__inference_dense_1_layer_call_and_return_conditional_losses_2878inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
рBн
*__inference_concatenate_layer_call_fn_2884inputs_0inputs_1"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ћBј
E__inference_concatenate_layer_call_and_return_conditional_losses_2891inputs_0inputs_1"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
аBЭ
&__inference_dense_2_layer_call_fn_2900inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ыBш
A__inference_dense_2_layer_call_and_return_conditional_losses_2910inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
аBЭ
&__inference_dense_3_layer_call_fn_2919inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ыBш
A__inference_dense_3_layer_call_and_return_conditional_losses_2929inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
аBЭ
&__inference_dense_4_layer_call_fn_2938inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ыBш
A__inference_dense_4_layer_call_and_return_conditional_losses_2948inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
аBЭ
&__inference_dense_5_layer_call_fn_2957inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ыBш
A__inference_dense_5_layer_call_and_return_conditional_losses_2968inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
R
о	variables
п	keras_api

рtotal

сcount"
_tf_keras_metric
c
т	variables
у	keras_api

фtotal

хcount
ц
_fn_kwargs"
_tf_keras_metric
0
р0
с1"
trackable_list_wrapper
.
о	variables"
_generic_user_object
:  (2total
:  (2count
0
ф0
х1"
trackable_list_wrapper
.
т	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapperИ
D__inference_Full_Model_layer_call_and_return_conditional_losses_1620яK67-.KLMNBC@AjkabvwtuГДЕЖЊЋЈЉежноыьѓєћќrЂo
hЂe
[X
%"
input_1џџџџџџџџџdt
/,
input_2џџџџџџџџџd
p

 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 И
D__inference_Full_Model_layer_call_and_return_conditional_losses_1748яK67-.KLMNC@BAjkabwtvuГДЕЖЋЈЊЉежноыьѓєћќrЂo
hЂe
[X
%"
input_1џџџџџџџџџdt
/,
input_2џџџџџџџџџd
p 

 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 
)__inference_Full_Model_layer_call_fn_1850фK67-.KLMNBC@AjkabvwtuГДЕЖЊЋЈЉежноыьѓєћќrЂo
hЂe
[X
%"
input_1џџџџџџџџџdt
/,
input_2џџџџџџџџџd
p

 
Њ "!
unknownџџџџџџџџџ
)__inference_Full_Model_layer_call_fn_1952фK67-.KLMNC@BAjkabwtvuГДЕЖЋЈЊЉежноыьѓєћќrЂo
hЂe
[X
%"
input_1џџџџџџџџџdt
/,
input_2џџџџџџџџџd
p 

 
Њ "!
unknownџџџџџџџџџ
__inference__wrapped_model_869ьK67-.KLMNC@BAjkabwtvuГДЕЖЋЈЊЉежноыьѓєћќjЂg
`Ђ]
[X
%"
input_1џџџџџџџџџdt
/,
input_2џџџџџџџџџd
Њ "1Њ.
,
dense_5!
dense_5џџџџџџџџџл
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2509vwtuDЂA
:Ђ7
-*
inputsџџџџџџџџџџџџџџџџџџ@
p

 
Њ "9Ђ6
/,
tensor_0џџџџџџџџџџџџџџџџџџ@
 л
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2529wtvuDЂA
:Ђ7
-*
inputsџџџџџџџџџџџџџџџџџџ@
p 

 
Њ "9Ђ6
/,
tensor_0џџџџџџџџџџџџџџџџџџ@
 Д
4__inference_batch_normalization_1_layer_call_fn_2462|vwtuDЂA
:Ђ7
-*
inputsџџџџџџџџџџџџџџџџџџ@
p

 
Њ ".+
unknownџџџџџџџџџџџџџџџџџџ@Д
4__inference_batch_normalization_1_layer_call_fn_2475|wtvuDЂA
:Ђ7
-*
inputsџџџџџџџџџџџџџџџџџџ@
p 

 
Њ ".+
unknownџџџџџџџџџџџџџџџџџџ@п
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2714ЊЋЈЉDЂA
:Ђ7
-*
inputsџџџџџџџџџџџџџџџџџџ@
p

 
Њ "9Ђ6
/,
tensor_0џџџџџџџџџџџџџџџџџџ@
 п
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2734ЋЈЊЉDЂA
:Ђ7
-*
inputsџџџџџџџџџџџџџџџџџџ@
p 

 
Њ "9Ђ6
/,
tensor_0џџџџџџџџџџџџџџџџџџ@
 Й
4__inference_batch_normalization_2_layer_call_fn_2667ЊЋЈЉDЂA
:Ђ7
-*
inputsџџџџџџџџџџџџџџџџџџ@
p

 
Њ ".+
unknownџџџџџџџџџџџџџџџџџџ@Й
4__inference_batch_normalization_2_layer_call_fn_2680ЋЈЊЉDЂA
:Ђ7
-*
inputsџџџџџџџџџџџџџџџџџџ@
p 

 
Њ ".+
unknownџџџџџџџџџџџџџџџџџџ@
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_2368ЛKLMN^Ђ[
TЂQ
GD
inputs8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@
p

 
Њ "SЂP
IF
tensor_08џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_2386ЛKLMN^Ђ[
TЂQ
GD
inputs8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@
p 

 
Њ "SЂP
IF
tensor_08џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 щ
4__inference_batch_normalization_3_layer_call_fn_2337АKLMN^Ђ[
TЂQ
GD
inputs8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@
p

 
Њ "HE
unknown8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@щ
4__inference_batch_normalization_3_layer_call_fn_2350АKLMN^Ђ[
TЂQ
GD
inputs8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@
p 

 
Њ "HE
unknown8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_2573О^Ђ[
TЂQ
GD
inputs8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@
p

 
Њ "SЂP
IF
tensor_08џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_2591О^Ђ[
TЂQ
GD
inputs8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@
p 

 
Њ "SЂP
IF
tensor_08џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 ь
4__inference_batch_normalization_4_layer_call_fn_2542Г^Ђ[
TЂQ
GD
inputs8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@
p

 
Њ "HE
unknown8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@ь
4__inference_batch_normalization_4_layer_call_fn_2555Г^Ђ[
TЂQ
GD
inputs8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@
p 

 
Њ "HE
unknown8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_2778ПГДЕЖ^Ђ[
TЂQ
GD
inputs8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@
p

 
Њ "SЂP
IF
tensor_08џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_2796ПГДЕЖ^Ђ[
TЂQ
GD
inputs8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@
p 

 
Њ "SЂP
IF
tensor_08џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 э
4__inference_batch_normalization_5_layer_call_fn_2747ДГДЕЖ^Ђ[
TЂQ
GD
inputs8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@
p

 
Њ "HE
unknown8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@э
4__inference_batch_normalization_5_layer_call_fn_2760ДГДЕЖ^Ђ[
TЂQ
GD
inputs8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@
p 

 
Њ "HE
unknown8џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ@й
M__inference_batch_normalization_layer_call_and_return_conditional_losses_2304BC@ADЂA
:Ђ7
-*
inputsџџџџџџџџџџџџџџџџџџ@
p

 
Њ "9Ђ6
/,
tensor_0џџџџџџџџџџџџџџџџџџ@
 й
M__inference_batch_normalization_layer_call_and_return_conditional_losses_2324C@BADЂA
:Ђ7
-*
inputsџџџџџџџџџџџџџџџџџџ@
p 

 
Њ "9Ђ6
/,
tensor_0џџџџџџџџџџџџџџџџџџ@
 В
2__inference_batch_normalization_layer_call_fn_2257|BC@ADЂA
:Ђ7
-*
inputsџџџџџџџџџџџџџџџџџџ@
p

 
Њ ".+
unknownџџџџџџџџџџџџџџџџџџ@В
2__inference_batch_normalization_layer_call_fn_2270|C@BADЂA
:Ђ7
-*
inputsџџџџџџџџџџџџџџџџџџ@
p 

 
Њ ".+
unknownџџџџџџџџџџџџџџџџџџ@д
E__inference_concatenate_layer_call_and_return_conditional_losses_2891ZЂW
PЂM
KH
"
inputs_0џџџџџџџџџ
"
inputs_1џџџџџџџџџ
Њ ",Ђ)
"
tensor_0џџџџџџџџџ

 ­
*__inference_concatenate_layer_call_fn_2884ZЂW
PЂM
KH
"
inputs_0џџџџџџџџџ
"
inputs_1џџџџџџџџџ
Њ "!
unknownџџџџџџџџџ
Б
B__inference_conv1d_1_layer_call_and_return_conditional_losses_2430kab3Ђ0
)Ђ&
$!
inputsџџџџџџџџџd@
Њ "0Ђ-
&#
tensor_0џџџџџџџџџd@
 
'__inference_conv1d_1_layer_call_fn_2415`ab3Ђ0
)Ђ&
$!
inputsџџџџџџџџџd@
Њ "%"
unknownџџџџџџџџџd@Г
B__inference_conv1d_2_layer_call_and_return_conditional_losses_2635m3Ђ0
)Ђ&
$!
inputsџџџџџџџџџd@
Њ "0Ђ-
&#
tensor_0џџџџџџџџџd@
 
'__inference_conv1d_2_layer_call_fn_2620b3Ђ0
)Ђ&
$!
inputsџџџџџџџџџd@
Њ "%"
unknownџџџџџџџџџd@Џ
@__inference_conv1d_layer_call_and_return_conditional_losses_2225k-.3Ђ0
)Ђ&
$!
inputsџџџџџџџџџdt
Њ "0Ђ-
&#
tensor_0џџџџџџџџџd@
 
%__inference_conv1d_layer_call_fn_2210`-.3Ђ0
)Ђ&
$!
inputsџџџџџџџџџdt
Њ "%"
unknownџџџџџџџџџd@Х
B__inference_conv3d_1_layer_call_and_return_conditional_losses_2449jk=Ђ:
3Ђ0
.+
inputsџџџџџџџџџd@
Њ ":Ђ7
0-
tensor_0џџџџџџџџџd@
 
'__inference_conv3d_1_layer_call_fn_2439tjk=Ђ:
3Ђ0
.+
inputsџџџџџџџџџd@
Њ "/,
unknownџџџџџџџџџd@Ш
B__inference_conv3d_2_layer_call_and_return_conditional_losses_2654=Ђ:
3Ђ0
.+
inputsџџџџџџџџџd@
Њ ":Ђ7
0-
tensor_0џџџџџџџџџd@
 Ё
'__inference_conv3d_2_layer_call_fn_2644v=Ђ:
3Ђ0
.+
inputsџџџџџџџџџd@
Њ "/,
unknownџџџџџџџџџd@У
@__inference_conv3d_layer_call_and_return_conditional_losses_224467=Ђ:
3Ђ0
.+
inputsџџџџџџџџџd
Њ ":Ђ7
0-
tensor_0џџџџџџџџџd@
 
%__inference_conv3d_layer_call_fn_2234t67=Ђ:
3Ђ0
.+
inputsџџџџџџџџџd
Њ "/,
unknownџџџџџџџџџd@Њ
A__inference_dense_1_layer_call_and_return_conditional_losses_2878eно/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 
&__inference_dense_1_layer_call_fn_2867Zно/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "!
unknownџџџџџџџџџЊ
A__inference_dense_2_layer_call_and_return_conditional_losses_2910eыь/Ђ,
%Ђ"
 
inputsџџџџџџџџџ

Њ ",Ђ)
"
tensor_0џџџџџџџџџ 
 
&__inference_dense_2_layer_call_fn_2900Zыь/Ђ,
%Ђ"
 
inputsџџџџџџџџџ

Њ "!
unknownџџџџџџџџџ Њ
A__inference_dense_3_layer_call_and_return_conditional_losses_2929eѓє/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 
&__inference_dense_3_layer_call_fn_2919Zѓє/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "!
unknownџџџџџџџџџЊ
A__inference_dense_4_layer_call_and_return_conditional_losses_2948eћќ/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 
&__inference_dense_4_layer_call_fn_2938Zћќ/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "!
unknownџџџџџџџџџЊ
A__inference_dense_5_layer_call_and_return_conditional_losses_2968e/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 
&__inference_dense_5_layer_call_fn_2957Z/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "!
unknownџџџџџџџџџЈ
?__inference_dense_layer_call_and_return_conditional_losses_2858eеж/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 
$__inference_dense_layer_call_fn_2847Zеж/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "!
unknownџџџџџџџџџй
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_2827IЂF
?Ђ<
63
inputs'џџџџџџџџџџџџџџџџџџџџџџџџџџџ

 
Њ "5Ђ2
+(
tensor_0џџџџџџџџџџџџџџџџџџ
 В
7__inference_global_average_pooling1d_layer_call_fn_2821wIЂF
?Ђ<
63
inputs'џџџџџџџџџџџџџџџџџџџџџџџџџџџ

 
Њ "*'
unknownџџџџџџџџџџџџџџџџџџя
R__inference_global_average_pooling3d_layer_call_and_return_conditional_losses_2838_Ђ\
UЂR
PM
inputsAџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "5Ђ2
+(
tensor_0џџџџџџџџџџџџџџџџџџ
 Щ
7__inference_global_average_pooling3d_layer_call_fn_2832_Ђ\
UЂR
PM
inputsAџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "*'
unknownџџџџџџџџџџџџџџџџџџЌ
A__inference_re_lu_1_layer_call_and_return_conditional_losses_2601g3Ђ0
)Ђ&
$!
inputsџџџџџџџџџd@
Њ "0Ђ-
&#
tensor_0џџџџџџџџџd@
 
&__inference_re_lu_1_layer_call_fn_2596\3Ђ0
)Ђ&
$!
inputsџџџџџџџџџd@
Њ "%"
unknownџџџџџџџџџd@Ќ
A__inference_re_lu_2_layer_call_and_return_conditional_losses_2806g3Ђ0
)Ђ&
$!
inputsџџџџџџџџџd@
Њ "0Ђ-
&#
tensor_0џџџџџџџџџd@
 
&__inference_re_lu_2_layer_call_fn_2801\3Ђ0
)Ђ&
$!
inputsџџџџџџџџџd@
Њ "%"
unknownџџџџџџџџџd@Р
A__inference_re_lu_3_layer_call_and_return_conditional_losses_2406{=Ђ:
3Ђ0
.+
inputsџџџџџџџџџd@
Њ ":Ђ7
0-
tensor_0џџџџџџџџџd@
 
&__inference_re_lu_3_layer_call_fn_2401p=Ђ:
3Ђ0
.+
inputsџџџџџџџџџd@
Њ "/,
unknownџџџџџџџџџd@Р
A__inference_re_lu_4_layer_call_and_return_conditional_losses_2611{=Ђ:
3Ђ0
.+
inputsџџџџџџџџџd@
Њ ":Ђ7
0-
tensor_0џџџџџџџџџd@
 
&__inference_re_lu_4_layer_call_fn_2606p=Ђ:
3Ђ0
.+
inputsџџџџџџџџџd@
Њ "/,
unknownџџџџџџџџџd@Р
A__inference_re_lu_5_layer_call_and_return_conditional_losses_2816{=Ђ:
3Ђ0
.+
inputsџџџџџџџџџd@
Њ ":Ђ7
0-
tensor_0џџџџџџџџџd@
 
&__inference_re_lu_5_layer_call_fn_2811p=Ђ:
3Ђ0
.+
inputsџџџџџџџџџd@
Њ "/,
unknownџџџџџџџџџd@Њ
?__inference_re_lu_layer_call_and_return_conditional_losses_2396g3Ђ0
)Ђ&
$!
inputsџџџџџџџџџd@
Њ "0Ђ-
&#
tensor_0џџџџџџџџџd@
 
$__inference_re_lu_layer_call_fn_2391\3Ђ0
)Ђ&
$!
inputsџџџџџџџџџd@
Њ "%"
unknownџџџџџџџџџd@Є
"__inference_signature_wrapper_2201§K67-.KLMNC@BAjkabwtvuГДЕЖЋЈЊЉежноыьѓєћќ{Ђx
Ђ 
qЊn
0
input_1%"
input_1џџџџџџџџџdt
:
input_2/,
input_2џџџџџџџџџd"1Њ.
,
dense_5!
dense_5џџџџџџџџџ