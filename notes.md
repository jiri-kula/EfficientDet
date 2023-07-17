batch 0
class prob at 7680 out of 12276

kde je v y_true class == 1

idxs = tf.where(y_true[0][...,4] == 1)

for idx in idxs: print(y_true[0][int(idx)])

box 1
x1 = 219.854336
x2 = 245.53984
y1 = 228.844288
y2 = 252.817408

box 2 
x1 = 186.034688
x2 = 213.860864
y1 = 231.412736
y2 = 255.385856

'positive_mask' ... zda maximální IOU přes libovolný box je větší než 0.5 - i.e. if any gt box hits anchor at respective index

'matched_gt_idx' ... selects from box that hits, you also need to look at positive, negative and ignore masks to get a grasp of the meaning of particular 'anchor box'

'label' from _encode_sample means [box_target, class_target], which is ['transform of anchor to match gt box', 'index of class (0..num_classes-1) or -1 if not positive box hit, or -2 if ignore mask at that anchor]

'So that what the net trains is anchor box offset + scale of each anchor to match the ground truth box.', which is the regression task.

---
pos = tf.where(label[...,-1] >= 0)
<tf.Tensor: shape=(8, 1), dtype=int64, numpy=
array([[8607],
       [8616],
       [8859],
       [8862],
       [8868],
       [8871],
       [8895],
       [8904]])>


anchor_boxes[8607]
<tf.Tensor: shape=(4,), dtype=float32, numpy=array([228., 236.,  32.,  32.], dtype=float32)>
classes
<tf.Tensor: shape=(2,), dtype=float32, numpy=array([0., 1.], dtype=float32)>
gt_boxes
array([[232.69708 , 240.83084 ,  25.685505,  23.97312 ],
       [199.94777 , 243.39929 ,  27.826176,  23.97312 ]], dtype=float32)
label[8607]
<tf.Tensor: shape=(5,), dtype=float32, numpy=
array([ 1.4678383,  1.5096378, -1.0990453, -1.4440136,  0.       ],
      dtype=float32)>