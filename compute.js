
function matrixToString(matrix) {
    const row = matrix[0];
    const col = matrix[1];
    matrix = matrix.slice(2);
    let result = "[";
    for (let i = 0; i < row; i++) {
        let rowString = "[";
        for (let j = 0; j < col; j++) {
            rowString += matrix[i * col + j] + ", ";
        }
        result += rowString.slice(0, -2) + "],\n";
    }
    return result.slice(0, -2) + "]\n";
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
    const l1 = Math.floor(Math.random() * 9 + 2), l2 = Math.floor(Math.random() * 9 + 2), l3 = Math.floor(Math.random() * 9 + 2);
    var matrixA = new Float32Array(l1 * l2 + 2);
    matrixA[0] = l1;
    matrixA[1] = l2;
    for (let i = 0; i < l1 * l2; i++) matrixA[i + 2] = Math.random() * 5;
    var matrixB = new Float32Array(l2 * l3 + 2);
    matrixB[0] = l2;
    matrixB[1] = l3;
    for (let i = 0; i < l2 * l3; i++) matrixB[i + 2] = Math.random() * 5;
    const resultMatrix = new Float32Array(l1 * l3 + 2);

    // === Load the WGSL shader code ===
    const shaderCode = await fetch('matmul.wgsl').then(res => res.text());
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
                                                    "Multiplication Result:\n" + resultString;
}

async function basicAttention() {

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
    const n = Math.floor(Math.random() * 19 + 2), d = Math.floor(Math.random() * 5 + 1);
    var matrixQ = new Float32Array(n * d + 2);
    matrixQ[0] = n;
    matrixQ[1] = d;
    for (let i = 0; i < n * d; i++) matrixQ[i + 2] = Math.random() * 5;
    var matrixK = new Float32Array(n * d + 2);
    matrixK[0] = n;
    matrixK[1] = d;
    for (let i = 0; i < n * d; i++) matrixK[i + 2] = Math.random() * 5;
    var matrixV = new Float32Array(n * d + 2);
    matrixV[0] = n;
    matrixV[1] = d;
    for (let i = 0; i < n * d; i++) matrixV[i + 2] = Math.random() * 5;
    const matrixS = new Float32Array(n * n + 2);
    const matrixT = new Float32Array(n * n + 2);
    const matrixO = new Float32Array(n * d + 2);

    // === Load the WGSL shader code ===
    const shaderCode = await fetch('basic-attention.wgsl').then(res => res.text());
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
                    type: "read-only-storage",
                },
            },
            {
                binding: 3,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: "storage",
                },
            },
            {
                binding: 4,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: "storage",
                },
            },
            {
                binding: 5,
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
            entryPoint: "attention",
        },
    });

    // === Create buffers ===
    // Staging buffers for efficient cpu -> gpu data transfer
    const stagingQBuffer = device.createBuffer({
        size: matrixQ.byteLength,
        usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC,
        mappedAtCreation: true,
    });
    new Float32Array(stagingQBuffer.getMappedRange()).set(matrixQ);
    stagingQBuffer.unmap();
    const stagingKBuffer = device.createBuffer({
        size: matrixK.byteLength,
        usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC,
        mappedAtCreation: true,
    });
    new Float32Array(stagingKBuffer.getMappedRange()).set(matrixK);
    stagingKBuffer.unmap();
    const stagingVBuffer = device.createBuffer({
        size: matrixV.byteLength,
        usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC,
        mappedAtCreation: true,
    });
    new Float32Array(stagingVBuffer.getMappedRange()).set(matrixV);
    stagingVBuffer.unmap();
    // GPU buffers needed for the shader
    const matrixQBuffer = device.createBuffer({
        size: matrixQ.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const matrixKBuffer = device.createBuffer({
        size: matrixK.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const matrixVBuffer = device.createBuffer({
        size: matrixV.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const matrixSBuffer = device.createBuffer({
        size: matrixS.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const matrixTBuffer = device.createBuffer({
        size: matrixT.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const matrixOBuffer = device.createBuffer({
        size: matrixO.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    // Output staging buffer
    const stagingOBuffer = device.createBuffer({
        size: matrixO.byteLength,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    // Bind group (only specify gpu buffers & not staging buffers)
    const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
            {
                binding: 0,
                resource: {
                    buffer: matrixQBuffer
                }
            },
            {
                binding: 1,
                resource: {
                    buffer: matrixKBuffer
                }
            },
            {
                binding: 2,
                resource: {
                    buffer: matrixVBuffer
                }
            },
            {
                binding: 3,
                resource: {
                    buffer: matrixSBuffer
                }
            },
            {
                binding: 4,
                resource: {
                    buffer: matrixTBuffer
                }
            },
            {
                binding: 5,
                resource: {
                    buffer: matrixOBuffer
                }
            },
        ],
    });
    
    // === Submit the commands to the GPU ===
    const commandEncoder = device.createCommandEncoder();
    // Inputs
    commandEncoder.copyBufferToBuffer(stagingQBuffer, 0, matrixQBuffer, 0, matrixQ.byteLength);
    commandEncoder.copyBufferToBuffer(stagingKBuffer, 0, matrixKBuffer, 0, matrixK.byteLength);
    commandEncoder.copyBufferToBuffer(stagingVBuffer, 0, matrixVBuffer, 0, matrixV.byteLength);
    // Compute
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    const workgroupCountX = Math.ceil(n / 8);
    const workgroupCountY = Math.ceil(Math.max(n, d) / 8);
    passEncoder.dispatchWorkgroups(workgroupCountX, workgroupCountY);
    passEncoder.end();
    // Output
    commandEncoder.copyBufferToBuffer(matrixOBuffer, 0, stagingOBuffer, 0, matrixO.byteLength);
    const commands = commandEncoder.finish();
    device.queue.submit([commands]);

    // === Read the result ===
    await stagingOBuffer.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(stagingOBuffer.getMappedRange());
    const copiedResult = Array.from(result);
    stagingOBuffer.unmap();

    // === Display the result ===
    const inputStringQ = matrixToString(matrixQ);
    const inputStringK = matrixToString(matrixK);
    const inputStringV = matrixToString(matrixV);
    const resultString = matrixToString(copiedResult);
    document.getElementById("result").textContent = "Input Q:\n" + inputStringQ + "\n" + 
                                                    "Input K:\n" + inputStringK + "\n" +
                                                    "Input V:\n" + inputStringV + "\n" +
                                                    "Attention Result:\n" + resultString;
}
