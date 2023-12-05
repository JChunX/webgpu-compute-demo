struct Matrix {
    size : vec2f,
    numbers: array<f32>,
}

@group(0) @binding(0) var<storage, read> qMatrix : Matrix;  // N x d
@group(0) @binding(1) var<storage, read> kMatrix : Matrix;  // N x d
@group(0) @binding(2) var<storage, read> vMatrix : Matrix;  // N x d
@group(0) @binding(3) var<storage, read_write> sMatrix : Matrix;  // N x N
@group(0) @binding(4) var<storage, read_write> tMatrix : Matrix;  // N x N
@group(0) @binding(5) var<storage, read_write> oMatrix : Matrix;  // N x d

// The compute shader entry point
@compute @workgroup_size(8, 8)
fn attention(
    @builtin(global_invocation_id) global_id : vec3<u32>
) {
    let n = u32(qMatrix.size.x);
    let d = u32(qMatrix.size.y);

    // s = q * k^T
    sMatrix.size = vec2(qMatrix.size.x, qMatrix.size.x);
    tMatrix.size = vec2(qMatrix.size.x, qMatrix.size.x);
    if (global_id.x < n && global_id.y < n) {
        var result = 0.0;
        for (var i = 0u; i < d; i = i + 1u) {
            result = result + qMatrix.numbers[global_id.x * d + i] * kMatrix.numbers[global_id.y * d + i];
        }
        sMatrix.numbers[global_id.x * n + global_id.y] = result;
        tMatrix.numbers[global_id.x * n + global_id.y] = result;
    }
    workgroupBarrier();

    // row-wise max
    for (var step = 1u; step < n; step = step * 2u) {
        if (global_id.x < n && global_id.y % (step * 2u) == 0 && global_id.y + step < n) {
            tMatrix.numbers[global_id.x * n + global_id.y] = max(tMatrix.numbers[global_id.x * n + global_id.y], tMatrix.numbers[global_id.x * n + global_id.y + step]);
        }
        workgroupBarrier();
    }

    // exponent
    if (global_id.x < n && global_id.y < n) {
        let temp = exp(sMatrix.numbers[global_id.x * n + global_id.y] - tMatrix.numbers[global_id.x * n]);
        sMatrix.numbers[global_id.x * n + global_id.y] = temp;
        tMatrix.numbers[global_id.x * n + global_id.y] = temp;
    }
    workgroupBarrier();

    // row-wise sum
    for (var step = 1u; step < n; step = step * 2u) {
        if (global_id.x < n && global_id.y % (step * 2u) == 0 && global_id.y + step < n) {
            tMatrix.numbers[global_id.x * n + global_id.y] = tMatrix.numbers[global_id.x * n + global_id.y] + tMatrix.numbers[global_id.x * n + global_id.y + step];
        }
        workgroupBarrier();
    }

    // softmax
    if (global_id.x < n && global_id.y < n) {
        sMatrix.numbers[global_id.x * n + global_id.y] = sMatrix.numbers[global_id.x * n + global_id.y] / tMatrix.numbers[global_id.x * n];
    }
    workgroupBarrier();

    // o = s * v
    oMatrix.size = vec2(qMatrix.size.x, qMatrix.size.y);
    if (global_id.x < n && global_id.y < d) {
        var result = 0.0;
        for (var i = 0u; i < n; i = i + 1u) {
            result = result + sMatrix.numbers[global_id.x * n + i] * vMatrix.numbers[i * d + global_id.y];
        }
        oMatrix.numbers[global_id.x * d + global_id.y] = result;
    }
    workgroupBarrier();

    //resultMatrix.size = vec2(firstMatrix.size.x, secondMatrix.size.y);

    //let resultCell = vec2(global_id.x, global_id.y);
    //var result = 0.0;
    //for (var i = 0u; i < u32(firstMatrix.size.y); i = i + 1u) {
    //    let a = i + resultCell.x * u32(firstMatrix.size.y);
    //    let b = resultCell.y + i * u32(secondMatrix.size.y);
    //    result = result + firstMatrix.numbers[a] * secondMatrix.numbers[b];
    //}

    //let index = resultCell.y + resultCell.x * u32(secondMatrix.size.y);
    //resultMatrix.numbers[index] = result;
    
}
