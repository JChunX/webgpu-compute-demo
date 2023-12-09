struct Matrix {
    size : vec2f,
    numbers: array<f32>,
}

@group(0) @binding(0) var<storage, read> sPatch : Matrix;  // Br x Bc
@group(0) @binding(1) var<storage, read> vPatch : Matrix;  // Bc x d
@group(0) @binding(2) var<storage, read_write> oPatch : Matrix;  // Br x d
@group(0) @binding(3) var<storage, read> mPatch : Matrix;  // Br x 1
@group(0) @binding(4) var<storage, read> lPatch : Matrix;  // Br x 1
@group(0) @binding(5) var<storage, read> mTemp : Matrix;  // Br x 1
@group(0) @binding(6) var<storage, read> mNew : Matrix;  // Br x 1
@group(0) @binding(7) var<storage, read> lNew : Matrix;  // Br x 1

// The compute shader entry point
@compute @workgroup_size(8, 8)
fn main(
    @builtin(global_invocation_id) global_id : vec3<u32>
) {
    let br = u32(sPatch.size.x);
    let bc = u32(sPatch.size.y);
    let d = u32(vPatch.size.y);

    if (global_id.x >= br || global_id.y >= d) {
        return;
    }

    var result = 0.0;
    for (var i = 0u; i < bc; i = i + 1u) {
        result = result + sPatch.numbers[global_id.x * bc + i] * vPatch.numbers[i * d + global_id.y];
    }
    oPatch.numbers[global_id.x * d + global_id.y] = (lPatch.numbers[global_id.x] * exp(mPatch.numbers[global_id.x] - mNew.numbers[global_id.x]) * oPatch.numbers[global_id.x * d + global_id.y] +
                                                     exp(mTemp.numbers[global_id.x] - mNew.numbers[global_id.x]) * result) / lNew.numbers[global_id.x];    
}
