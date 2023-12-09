struct Matrix {
    size : vec2f,
    numbers: array<f32>,
}

@group(0) @binding(0) var<storage, read_write> sPatch : Matrix;  // Br x Bc
@group(0) @binding(1) var<storage, read> mPatch : Matrix;  // Br x 1
@group(0) @binding(2) var<storage, read> lPatch : Matrix;  // Br x 1
@group(0) @binding(3) var<storage, read_write> mTemp : Matrix;  // Br x 1
@group(0) @binding(4) var<storage, read_write> mNew : Matrix;  // Br x 1
@group(0) @binding(5) var<storage, read_write> lNew : Matrix;  // Br x 1

// The compute shader entry point
@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id) global_id : vec3<u32>
) {
    let br = u32(sPatch.size.x);
    let bc = u32(sPatch.size.y);

    if (global_id.x >= br) {
        return;
    }
    mTemp.size = vec2(sPatch.size.x, 1);
    mNew.size = vec2(sPatch.size.x, 1);
    lNew.size = vec2(sPatch.size.x, 1);

    var rowMax = sPatch.numbers[global_id.x * bc];
    for (var i = 1u; i < bc; i = i + 1u) {
        rowMax = max(rowMax, sPatch.numbers[global_id.x * bc + i]);
    }
    mTemp.numbers[global_id.x] = rowMax;
    
    var rowSum = 0.0;
    for (var i = 0u; i < bc; i = i + 1u) {
        sPatch.numbers[global_id.x * bc + i] = exp(sPatch.numbers[global_id.x * bc + i] - rowMax);
        rowSum = rowSum + sPatch.numbers[global_id.x * bc + i];
    }
    mNew.numbers[global_id.x] = max(mPatch.numbers[global_id.x], rowMax);
    lNew.numbers[global_id.x] = exp(mPatch.numbers[global_id.x] - mNew.numbers[global_id.x]) * lPatch.numbers[global_id.x] +
                                exp(rowMax - mNew.numbers[global_id.x]) * rowSum;
}
