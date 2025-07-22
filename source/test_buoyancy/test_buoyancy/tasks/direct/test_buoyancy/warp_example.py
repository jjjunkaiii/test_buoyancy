import warp as wp
import numpy as np

np.random.seed(7)
c = np.random.rand(10)
d = wp.array(c, dtype=float, device="cuda:0")
print(d.numpy())
e = np.random.rand(10)

@wp.kernel
def main(arr: wp.array(dtype=float)):
    tid = wp.tid()
    print(tid)
    print("aaaaaaaaa")
    print(arr[tid*0+1])

wp.launch(
            kernel=main,
            dim=3, 
            inputs=[d], 
            outputs=[],
            device="cuda:0")