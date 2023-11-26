
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

async function multiplyMatrices() {

    // === Get a GPU device ===
    if (!navigator.gpu) throw Error("WebGPU not supported.");
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw Error("Could not request WebGPU adapter.");
    const deviceDescriptor = {
        nonGuaranteedLimits: {
            "maxMemoryPerDevice": 8 * 1024 * 1024 * 1024 // 8GB in bytes
        }
    };
    const device = await adapter.requestDevice(deviceDescriptor);
    if (!device) throw Error("Could not request WebGPU logical device.");

    // === Define matrices ===
    const matrixA = new Float32Array([3, 3, // row, col info
                                      1, 2, 3, 
                                      4, 5, 6, 
                                      7, 8, 9]); 
    const matrixB = new Float32Array([3, 3, // row, col info
                                      9, 8, 7, 
                                      6, 5, 4, 
                                      3, 2, 1]); 
    const resultMatrix = new Float32Array(11); // Result 3x3 matrix

    // === Load the WGSL shader code ===
    const shaderCode = await fetch('kernel.wgsl').then(res => res.text());
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
        size: matrixA.byteLength,
        usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC,
        mappedAtCreation: true,
    });
    new Float32Array(stagingABuffer.getMappedRange()).set(matrixA);
    stagingABuffer.unmap();
    const stagingBBuffer = device.createBuffer({
        size: matrixB.byteLength,
        usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC,
        mappedAtCreation: true,
    });
    new Float32Array(stagingBBuffer.getMappedRange()).set(matrixB);
    stagingBBuffer.unmap();
    // GPU buffers needed for the shader
    const matrixABuffer = device.createBuffer({
        size: matrixA.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const matrixBBuffer = device.createBuffer({
        size: matrixB.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const resultMatrixBuffer = device.createBuffer({
        size: resultMatrix.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    // Output staging buffer
    const stagingResultBuffer = device.createBuffer({
        size: resultMatrix.byteLength,
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
    commandEncoder.copyBufferToBuffer(stagingABuffer, 0, matrixABuffer, 0, matrixA.byteLength);
    commandEncoder.copyBufferToBuffer(stagingBBuffer, 0, matrixBBuffer, 0, matrixB.byteLength);
    // Compute
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    const workgroupCountX = Math.ceil(matrixA[0] / 8);
    const workgroupCountY = Math.ceil(matrixB[1] / 8);
    passEncoder.dispatchWorkgroups(workgroupCountX, workgroupCountY);
    passEncoder.end();
    // Output
    commandEncoder.copyBufferToBuffer(resultMatrixBuffer, 0, stagingResultBuffer, 0, resultMatrix.byteLength);
    const commands = commandEncoder.finish();
    device.queue.submit([commands]);

    // === Read the result ===
    await stagingResultBuffer.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(stagingResultBuffer.getMappedRange());
    const copiedResult = Array.from(result);
    stagingResultBuffer.unmap();

    // === Display the result ===
    const inputStringA = matrixToString(matrixA);
    const inputStringB = matrixToString(matrixB);
    const resultString = matrixToString(copiedResult);
    document.getElementById("result").textContent = "Input A:\n" + inputStringA + "\n" + 
                                                    "Input B:\n" + inputStringB + "\n" +
                                                    "Result:\n" + resultString;
}
