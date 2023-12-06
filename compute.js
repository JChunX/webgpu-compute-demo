
function matrixToString(matrix) {
    const row = matrix[0];
    const col = matrix[1];
    matrix = matrix.slice(2);
    let result = "";
    for (let i = 0; i < row; i++) {
        let rowString = "";
        for (let j = 0; j < col; j++) {
            rowString += matrix[i * col + j] + " ";
        }
        result += rowString + "\n";
    }
    return result;
}

async function getDevice(maxMemory) {
    if (!navigator.gpu) throw Error("WebGPU not supported.");
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw Error("Could not request WebGPU adapter.");
    const requiredLimits = {};
    requiredLimits.maxStorageBufferBindingSize = maxMemory;
    requiredLimits.maxBufferSize = maxMemory;
    const deviceDescriptor = {
        requiredLimits,
        requiredFeatures: ["shader-f16"],
    };
    const device = await adapter.requestDevice(deviceDescriptor);
    if (!device) throw Error("Could not request WebGPU logical device.");
    return device;
}

function float32ToFloat16(float32) {
    let float32View = new Float32Array(1);
    let int32View = new Int32Array(float32View.buffer);

    float32View[0] = float32;
    let f32 = int32View[0];

    let f16 = 0;
    switch ((f32 >> 23) & 0xff) {
        case 0x00: // Zero / Denormal
            f16 = (f32 >> 16) & 0x8000;
            break;
        case 0xff: // Infinity / NaN
            f16 = ((f32 >> 16) & 0x8000) | 0x7c00 | (f32 & 0x007fffff ? 0x0200 : 0x0);
            break;
        default: // Normal
            f16 = ((f32 >> 16) & 0x8000) | ((((f32 & 0x7fffffff) >> 13) - 0x1c000) & 0x7fff);
            break;
    }
    return f16;
}

function convertMatrixToFloat16(matrix) {
    let uint16Array = new Uint16Array(matrix.length);
    for (let i = 0; i < matrix.length; i++) {
        uint16Array[i] = float32ToFloat16(matrix[i]);
    }
    return uint16Array;
}

function float16ToFloat32(uint16) {
    let t1 = uint16 & 0x7fff;         // Non-sign bits
    let t2 = uint16 & 0x8000;         // Sign bit
    let t3 = uint16 & 0x7c00;         // Exponent

    t1 <<= 13;                        // Align mantissa on MSB
    t2 <<= 16;                        // Shift sign bit into position

    t1 += 0x38000000;                 // Adjust bias

    t1 = (t3 === 0 ? 0 : t1);         // Denormals-as-zero
    t1 |= t2;                         // Re-insert sign bit

    let float32View = new Float32Array(1);
    let int32View = new Int32Array(float32View.buffer);
    int32View[0] = t1;
    return float32View[0];
}

function convertMatrixToFloat32(uint16Matrix) {
    let float32Array = new Float32Array(uint16Matrix.length);
    for (let i = 0; i < uint16Matrix.length; i++) {
        float32Array[i] = float16ToFloat32(uint16Matrix[i]);
    }
    return float32Array;
}

async function multiplyMatrices() {

    // === Get a GPU device ===
    device = await getDevice(1 * 1024 * 1024 * 1024); // 1 GB

    // === Define matrices ===
    const matrixA = new Float32Array([3, 3, // row, col info
                                      1, 2, 3, 
                                      4, 5, 6, 
                                      7, 8, 9]); 
    const matrixB = new Float32Array([3, 3, // row, col info
                                      9, 8, 7, 
                                      6, 5, 4, 
                                      3, 2, 1]); 
    const resultMatrix = new Uint16Array(3 * 3 + 2); // row, col info

    const matrixA16 = convertMatrixToFloat16(matrixA);
    const matrixB16 = convertMatrixToFloat16(matrixB);

    // === Load the WGSL shader code ===
    const shaderCode = await fetch('f16_kernel.wgsl').then(res => res.text());
    const shaderModule = device.createShaderModule({ code: shaderCode });
    // Bind group layout (see actual bind group & buffer creation below)
    const bindGroupLayout = device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: "read-only-storage",
                },
            },
            {
                binding: 1,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: "read-only-storage",
                },
            },
            {
                binding: 2,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: "storage",
                },
            },
        ],
    });
    const pipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({ 
            bindGroupLayouts: [bindGroupLayout] 
        }),
        compute: {
            module: shaderModule,
            entryPoint: "matmul",
        },
    });

    // === Create buffers ===
    // Staging buffers for efficient cpu -> gpu data transfer
    const stagingABuffer = device.createBuffer({
        size: matrixA16.byteLength + matrixB16.byteLength % 4,
        usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC,
        mappedAtCreation: true,
    });
    new Uint16Array(stagingABuffer.getMappedRange()).set(matrixA16);
    stagingABuffer.unmap();
    const stagingBBuffer = device.createBuffer({
        size: matrixB16.byteLength + matrixB16.byteLength % 4,
        usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC,
        mappedAtCreation: true,
    });
    new Uint16Array(stagingBBuffer.getMappedRange()).set(matrixB16);
    stagingBBuffer.unmap();
    // GPU buffers needed for the shader
    const matrixABuffer = device.createBuffer({
        size: matrixA16.byteLength + matrixB16.byteLength % 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const matrixBBuffer = device.createBuffer({
        size: matrixB16.byteLength + matrixB16.byteLength % 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const resultMatrixBuffer = device.createBuffer({
        size: resultMatrix.byteLength + resultMatrix.byteLength % 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    // Output staging buffer
    const stagingResultBuffer = device.createBuffer({
        size: resultMatrix.byteLength + resultMatrix.byteLength % 4,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    // Bind group (only specify gpu buffers & not staging buffers)
    const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
            {
                binding: 0,
                resource: {
                    buffer: matrixABuffer
                }
            },
            {
                binding: 1,
                resource: {
                    buffer: matrixBBuffer
                }
            },
            {
                binding: 2,
                resource: {
                    buffer: resultMatrixBuffer
                }
            }
        ],
    });
    
    // === Submit the commands to the GPU ===
    const commandEncoder = device.createCommandEncoder();
    // Inputs
    commandEncoder.copyBufferToBuffer(stagingABuffer, 0, matrixABuffer, 0, matrixA16.byteLength + matrixB16.byteLength % 4);
    commandEncoder.copyBufferToBuffer(stagingBBuffer, 0, matrixBBuffer, 0, matrixB16.byteLength + matrixB16.byteLength % 4);
    // Compute
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    const workgroupCountX = Math.ceil(matrixA[0] / 8);
    const workgroupCountY = Math.ceil(matrixB[1] / 8);
    passEncoder.dispatchWorkgroups(workgroupCountX, workgroupCountY);
    passEncoder.end();
    // Output
    commandEncoder.copyBufferToBuffer(resultMatrixBuffer, 0, stagingResultBuffer, 0, resultMatrix.byteLength + resultMatrix.byteLength % 4);
    const commands = commandEncoder.finish();
    device.queue.submit([commands]);

    // === Read the result ===
    await stagingResultBuffer.mapAsync(GPUMapMode.READ);
    const result = new Uint16Array(stagingResultBuffer.getMappedRange());
    const copiedResult = Array.from(result);
    stagingResultBuffer.unmap();

    // === Display the result ===
    const inputStringA = matrixToString(matrixA);
    const inputStringB = matrixToString(matrixB);
    const resultString = matrixToString(convertMatrixToFloat32(copiedResult));
    document.getElementById("result").textContent = "Input A:\n" + inputStringA + "\n" + 
                                                    "Input B:\n" + inputStringB + "\n" +
                                                    "Result:\n" + resultString;
}
