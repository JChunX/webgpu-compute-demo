
function matrixToString(matrix, precision=4) {
    const row = matrix[0];
    const col = matrix[1];
    matrix = matrix.slice(2);
    let result = "[";
    for (let i = 0; i < row; i++) {
        let rowString = "[";
        for (let j = 0; j < col; j++) {
            rowString += Math.floor(matrix[i * col + j] * Math.pow(10, precision)) / Math.pow(10, precision) + ", ";
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
    const l1 = Math.floor(Math.random() * 9 + 2), l2 = Math.floor(Math.random() * 9 + 2), l3 = Math.floor(Math.random() * 9 + 2), l4 = Math.floor(Math.random() * 9 + 2);
    var matrixA = new Float32Array(l1 * l2 + 2);
    matrixA[0] = l1; matrixA[1] = l2;
    for (let i = 0; i < l1 * l2; i++) matrixA[i + 2] = Math.random() * 5;
    var matrixB = new Float32Array(l2 * l3 + 2);
    matrixB[0] = l2; matrixB[1] = l3;
    for (let i = 0; i < l2 * l3; i++) matrixB[i + 2] = Math.random() * 5;
    var matrixC = new Float32Array(l3 * l4 + 2);
    matrixC[0] = l3; matrixC[1] = l4;
    for (let i = 0; i < l3 * l4; i++) matrixC[i + 2] = Math.random() * 5;

    // === Load the WGSL shader code ===
    const shaderCode = await fetch('matmul.wgsl').then(res => res.text());
    const shaderModule = device.createShaderModule({ code: shaderCode });
    // Bind group layout (see actual bind group & buffer creation below)
    const bindGroupLayout = device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "read-only-storage" },
            },
            {
                binding: 1,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "read-only-storage" },
            },
            {
                binding: 2,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "storage" },
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
    const stagingCBuffer = device.createBuffer({
        size: matrixC.byteLength,
        usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC,
        mappedAtCreation: true,
    });
    new Float32Array(stagingCBuffer.getMappedRange()).set(matrixC);
    stagingCBuffer.unmap();
    // GPU buffers needed for the shader
    const matrixABuffer = device.createBuffer({
        size: matrixA.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const matrixBBuffer = device.createBuffer({
        size: matrixB.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const matrixCBuffer = device.createBuffer({
        size: matrixC.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const matrixABBuffer = device.createBuffer({
        size: (l1 * l3 + 2) * Float32Array.BYTES_PER_ELEMENT,
        usage: GPUBufferUsage.STORAGE,
    });
    const resultMatrixBuffer = device.createBuffer({
        size: (l1 * l4 + 2) * Float32Array.BYTES_PER_ELEMENT,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    // Output staging buffer
    const stagingResultBuffer = device.createBuffer({
        size: (l1 * l4 + 2) * Float32Array.BYTES_PER_ELEMENT,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    // Bind group (only specify gpu buffers & not staging buffers)
    const bindGroup1 = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
            {
                binding: 0,
                resource: { buffer: matrixABuffer }
            },
            {
                binding: 1,
                resource: { buffer: matrixBBuffer }
            },
            {
                binding: 2,
                resource: { buffer: matrixABBuffer }
            }
        ],
    });
    const bindGroup2 = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
            {
                binding: 0,
                resource: { buffer: matrixABBuffer }
            },
            {
                binding: 1,
                resource: { buffer: matrixCBuffer }
            },
            {
                binding: 2,
                resource: { buffer: resultMatrixBuffer }
            }
        ],
    });
    
    // === Submit the commands to the GPU ===
    const commandEncoder = device.createCommandEncoder();
    // Inputs
    commandEncoder.copyBufferToBuffer(stagingABuffer, 0, matrixABuffer, 0, matrixA.byteLength);
    commandEncoder.copyBufferToBuffer(stagingBBuffer, 0, matrixBBuffer, 0, matrixB.byteLength);
    commandEncoder.copyBufferToBuffer(stagingCBuffer, 0, matrixCBuffer, 0, matrixC.byteLength);
    // Compute
    const passEncoder1 = commandEncoder.beginComputePass();
    passEncoder1.setPipeline(pipeline);
    passEncoder1.setBindGroup(0, bindGroup1);
    passEncoder1.dispatchWorkgroups(Math.ceil(l1 / 8), Math.ceil(l3 / 8));
    passEncoder1.end();
    const passEncoder2 = commandEncoder.beginComputePass();
    passEncoder2.setPipeline(pipeline);
    passEncoder2.setBindGroup(0, bindGroup2);
    passEncoder2.dispatchWorkgroups(Math.ceil(l1 / 8), Math.ceil(l4 / 8));
    passEncoder2.end();
    // Output
    commandEncoder.copyBufferToBuffer(resultMatrixBuffer, 0, stagingResultBuffer, 0, (l1 * l4 + 2) * Float32Array.BYTES_PER_ELEMENT);
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
    const inputStringC = matrixToString(matrixC);
    const resultString = matrixToString(copiedResult);
    document.getElementById("result").innerHTML = "<table><tr><td>Input A:\n" + inputStringA + "</td>" + 
                                                  "<td>Input B:\n" + inputStringB + "</td>" +
                                                  "<td>Input C:\n" + inputStringC + "</td></tr></table>" +
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
    matrixQ[0] = n; matrixQ[1] = d;
    for (let i = 0; i < n * d; i++) matrixQ[i + 2] = Math.random() * 5;
    var matrixK = new Float32Array(n * d + 2);
    matrixK[0] = n; matrixK[1] = d;
    for (let i = 0; i < n * d; i++) matrixK[i + 2] = Math.random() * 5;
    var matrixV = new Float32Array(n * d + 2);
    matrixV[0] = n; matrixV[1] = d;
    for (let i = 0; i < n * d; i++) matrixV[i + 2] = Math.random() * 5;

    // === Load the WGSL shader code ===
    const shaderCode = await fetch('basic-attention.wgsl').then(res => res.text());
    const shaderModule = device.createShaderModule({ code: shaderCode });
    // Bind group layout (see actual bind group & buffer creation below)
    const bindGroupLayout = device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "read-only-storage" },
            },
            {
                binding: 1,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "read-only-storage" },
            },
            {
                binding: 2,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "read-only-storage" },
            },
            {
                binding: 3,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "storage" },
            },
            {
                binding: 4,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "storage" },
            },
            {
                binding: 5,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "storage" },
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
        size: (n * n + 2) * Float32Array.BYTES_PER_ELEMENT,
        usage: GPUBufferUsage.STORAGE,
    });
    const matrixTBuffer = device.createBuffer({
        size: (n * n + 2) * Float32Array.BYTES_PER_ELEMENT,
        usage: GPUBufferUsage.STORAGE,
    });
    const matrixOBuffer = device.createBuffer({
        size: (n * d + 2) * Float32Array.BYTES_PER_ELEMENT,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    // Output staging buffer
    const stagingOBuffer = device.createBuffer({
        size: (n * d + 2) * Float32Array.BYTES_PER_ELEMENT,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    // Bind group (only specify gpu buffers & not staging buffers)
    const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
            {
                binding: 0,
                resource: { buffer: matrixQBuffer }
            },
            {
                binding: 1,
                resource: { buffer: matrixKBuffer }
            },
            {
                binding: 2,
                resource: { buffer: matrixVBuffer }
            },
            {
                binding: 3,
                resource: { buffer: matrixSBuffer }
            },
            {
                binding: 4,
                resource: { buffer: matrixTBuffer }
            },
            {
                binding: 5,
                resource: { buffer: matrixOBuffer }
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
    commandEncoder.copyBufferToBuffer(matrixOBuffer, 0, stagingOBuffer, 0, (n * d + 2) * Float32Array.BYTES_PER_ELEMENT);
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
    document.getElementById("result").innerHTML = "<table><tr><td>Input Q:\n" + inputStringQ + "</td>" + 
                                                  "<td>Input K:\n" + inputStringK + "</td>" +
                                                  "<td>Input V:\n" + inputStringV + "</td></tr></table>" +
                                                  "Attention Result:\n" + resultString;
}

async function flashAttention() {

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
    //const n = Math.floor(Math.random() * 19 + 2), d = Math.floor(Math.random() * 5 + 1);
    const n = 500, d = 64;
    var matrixQ = new Float32Array(n * d + 2);
    matrixQ[0] = n; matrixQ[1] = d;
    for (let i = 0; i < n * d; i++) matrixQ[i + 2] = Math.random() * 5;
    var matrixK = new Float32Array(n * d + 2);
    matrixK[0] = n; matrixK[1] = d;
    for (let i = 0; i < n * d; i++) matrixK[i + 2] = Math.random() * 5;
    var matrixV = new Float32Array(n * d + 2);
    matrixV[0] = n; matrixV[1] = d;
    for (let i = 0; i < n * d; i++) matrixV[i + 2] = Math.random() * 5;
    // var matrixO = new Float32Array(n * d + 2);
    // matrixO[0] = n; matrixO[1] = d;
    // for (let i = 0; i < n * d; i++) matrixO[i + 2] = 0;
    var matrixL = new Float32Array(n + 2);
    matrixL[0] = n; matrixL[1] = 1;
    for (let i = 0; i < n; i++) matrixL[i + 2] = 0;
    var matrixM = new Float32Array(n + 2);
    matrixM[0] = n; matrixM[1] = 1;
    for (let i = 0; i < n; i++) matrixM[i + 2] = Number.NEGATIVE_INFINITY;

    const m = 100 * 1024; // 100KB SRAM
    const bc = Math.ceil(m / (Float32Array.BYTES_PER_ELEMENT * d)), br = Math.min(bc, d);
    var patchQ = new Float32Array(br * d + 2);
    patchQ[0] = br; patchQ[1] = d;
    var patchK = new Float32Array(bc * d + 2);
    patchK[0] = bc; patchK[1] = d;
    var patchV = new Float32Array(bc * d + 2);
    patchV[0] = bc; patchV[1] = d;

    // === Load the WGSL shader code ===
    const shaderCode1 = await fetch('matmultrans.wgsl').then(res => res.text());
    const shaderModule1 = device.createShaderModule({ code: shaderCode1 });
    const shaderCode2 = await fetch('fa-rowwise.wgsl').then(res => res.text());
    const shaderModule2 = device.createShaderModule({ code: shaderCode2 });
    const shaderCode3 = await fetch('fa-output.wgsl').then(res => res.text());
    const shaderModule3 = device.createShaderModule({ code: shaderCode3 });
    // Bind group layout (see actual bind group & buffer creation below)
    const bindGroupLayout1 = device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "read-only-storage" },
            },
            {
                binding: 1,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "read-only-storage" },
            },
            {
                binding: 2,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "storage" },
            },
        ],
    });
    const bindGroupLayout2 = device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "storage" },
            },
            {
                binding: 1,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "read-only-storage" },
            },
            {
                binding: 2,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "read-only-storage" },
            },
            {
                binding: 3,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "storage" },
            },
            {
                binding: 4,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "storage" },
            },
            {
                binding: 5,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "storage" },
            },
        ],
    });
    const bindGroupLayout3 = device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "read-only-storage" },
            },
            {
                binding: 1,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "read-only-storage" },
            },
            {
                binding: 2,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "storage" },
            },
            {
                binding: 3,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "read-only-storage" },
            },
            {
                binding: 4,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "read-only-storage" },
            },
            {
                binding: 5,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "read-only-storage" },
            },
            {
                binding: 6,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "read-only-storage" },
            },
            {
                binding: 7,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "read-only-storage" },
            },
        ],
    });
    const pipeline1 = device.createComputePipeline({
        layout: device.createPipelineLayout({ 
            bindGroupLayouts: [bindGroupLayout1] 
        }),
        compute: {
            module: shaderModule1,
            entryPoint: "matmul",
        },
    });
    const pipeline2 = device.createComputePipeline({
        layout: device.createPipelineLayout({ 
            bindGroupLayouts: [bindGroupLayout2] 
        }),
        compute: {
            module: shaderModule2,
            entryPoint: "main",
        },
    });
    const pipeline3 = device.createComputePipeline({
        layout: device.createPipelineLayout({ 
            bindGroupLayouts: [bindGroupLayout3] 
        }),
        compute: {
            module: shaderModule3,
            entryPoint: "main",
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
    const stagingLBuffer = device.createBuffer({
        size: matrixL.byteLength,
        usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC,
        mappedAtCreation: true,
    });
    new Float32Array(stagingLBuffer.getMappedRange()).set(matrixL);
    stagingLBuffer.unmap();
    const stagingMBuffer = device.createBuffer({
        size: matrixM.byteLength,
        usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC,
        mappedAtCreation: true,
    });
    new Float32Array(stagingMBuffer.getMappedRange()).set(matrixM);
    stagingMBuffer.unmap();

    const stagingQPatchBuffer = device.createBuffer({
        size: patchQ.byteLength,
        usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC,
        mappedAtCreation: true,
    });
    new Float32Array(stagingQPatchBuffer.getMappedRange()).set(patchQ);
    stagingQPatchBuffer.unmap();
    const stagingKPatchBuffer = device.createBuffer({
        size: patchK.byteLength,
        usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC,
        mappedAtCreation: true,
    });
    new Float32Array(stagingKPatchBuffer.getMappedRange()).set(patchK);
    stagingKPatchBuffer.unmap();
    const stagingVPatchBuffer = device.createBuffer({
        size: patchV.byteLength,
        usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC,
        mappedAtCreation: true,
    });
    new Float32Array(stagingVPatchBuffer.getMappedRange()).set(patchV);
    stagingVPatchBuffer.unmap();

    // GPU buffers needed for the shader
    const matrixQBuffer = device.createBuffer({
        size: matrixQ.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const matrixKBuffer = device.createBuffer({
        size: matrixK.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const matrixVBuffer = device.createBuffer({
        size: matrixV.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const matrixOBuffer = device.createBuffer({
        size: (n * d + 2) * Float32Array.BYTES_PER_ELEMENT,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const matrixLBuffer = device.createBuffer({
        size: matrixL.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const matrixMBuffer = device.createBuffer({
        size: matrixM.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    const patchQBuffer = device.createBuffer({
        size: patchQ.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const patchKBuffer = device.createBuffer({
        size: patchK.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const patchVBuffer = device.createBuffer({
        size: patchV.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const patchOBuffer = device.createBuffer({
        size: (br * d + 2) * Float32Array.BYTES_PER_ELEMENT,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const patchLBuffer = device.createBuffer({
        size: (br + 2) * Float32Array.BYTES_PER_ELEMENT,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const patchMBuffer = device.createBuffer({
        size: (br + 2) * Float32Array.BYTES_PER_ELEMENT,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const patchSBuffer = device.createBuffer({
        size: (br * bc + 2) * Float32Array.BYTES_PER_ELEMENT,
        usage: GPUBufferUsage.STORAGE,
    });
    const newLBuffer = device.createBuffer({
        size: (br + 2) * Float32Array.BYTES_PER_ELEMENT,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const newMBuffer = device.createBuffer({
        size: (br + 2) * Float32Array.BYTES_PER_ELEMENT,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const tempMBuffer = device.createBuffer({
        size: (br + 2) * Float32Array.BYTES_PER_ELEMENT,
        usage: GPUBufferUsage.STORAGE,
    });

    // Output staging buffer
    const stagingOBuffer = device.createBuffer({
        size: (n * d + 2) * Float32Array.BYTES_PER_ELEMENT,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    // Bind group (only specify gpu buffers & not staging buffers)
    const bindGroup1 = device.createBindGroup({
        layout: bindGroupLayout1,
        entries: [
            {
                binding: 0,
                resource: { buffer: patchQBuffer }
            },
            {
                binding: 1,
                resource: { buffer: patchKBuffer }
            },
            {
                binding: 2,
                resource: { buffer: patchSBuffer }
            },
        ],
    });
    const bindGroup2 = device.createBindGroup({
        layout: bindGroupLayout2,
        entries: [
            {
                binding: 0,
                resource: { buffer: patchSBuffer }
            },
            {
                binding: 1,
                resource: { buffer: patchMBuffer }
            },
            {
                binding: 2,
                resource: { buffer: patchLBuffer }
            },
            {
                binding: 3,
                resource: { buffer: tempMBuffer }
            },
            {
                binding: 4,
                resource: { buffer: newMBuffer }
            },
            {
                binding: 5,
                resource: { buffer: newLBuffer }
            },
        ],
    });
    const bindGroup3 = device.createBindGroup({
        layout: bindGroupLayout3,
        entries: [
            {
                binding: 0,
                resource: { buffer: patchSBuffer }
            },
            {
                binding: 1,
                resource: { buffer: patchVBuffer }
            },
            {
                binding: 2,
                resource: { buffer: patchOBuffer }
            },
            {
                binding: 3,
                resource: { buffer: patchMBuffer }
            },
            {
                binding: 4,
                resource: { buffer: patchLBuffer }
            },
            {
                binding: 5,
                resource: { buffer: tempMBuffer }
            },
            {
                binding: 6,
                resource: { buffer: newMBuffer }
            },
            {
                binding: 7,
                resource: { buffer: newLBuffer }
            },
        ],
    });
    
    // === Submit the commands to the GPU ===
    const commandEncoder = device.createCommandEncoder();
    // Inputs
    commandEncoder.copyBufferToBuffer(stagingQBuffer, 0, matrixQBuffer, 0, matrixQ.byteLength);
    commandEncoder.copyBufferToBuffer(stagingKBuffer, 0, matrixKBuffer, 0, matrixK.byteLength);
    commandEncoder.copyBufferToBuffer(stagingVBuffer, 0, matrixVBuffer, 0, matrixV.byteLength);
    commandEncoder.copyBufferToBuffer(stagingLBuffer, 0, matrixLBuffer, 0, matrixL.byteLength);
    commandEncoder.copyBufferToBuffer(stagingMBuffer, 0, matrixMBuffer, 0, matrixM.byteLength);
    commandEncoder.copyBufferToBuffer(stagingQPatchBuffer, 0, patchQBuffer, 0, patchQ.byteLength);
    commandEncoder.copyBufferToBuffer(stagingKPatchBuffer, 0, patchKBuffer, 0, patchK.byteLength);
    commandEncoder.copyBufferToBuffer(stagingVPatchBuffer, 0, patchVBuffer, 0, patchV.byteLength);
    // Compute
    for (let j = 0; j < Math.ceil(n / bc); j++) {
        commandEncoder.copyBufferToBuffer(matrixKBuffer, (bc * j * d + 2) * Float32Array.BYTES_PER_ELEMENT,
                                          patchKBuffer, 2 * Float32Array.BYTES_PER_ELEMENT,
                                          Math.min(bc, n - j * bc) * d * Float32Array.BYTES_PER_ELEMENT);
        commandEncoder.copyBufferToBuffer(matrixVBuffer, (bc * j * d + 2) * Float32Array.BYTES_PER_ELEMENT,
                                          patchVBuffer, 2 * Float32Array.BYTES_PER_ELEMENT,
                                          Math.min(bc, n - j * bc) * d * Float32Array.BYTES_PER_ELEMENT);
        for (let i = 0; i < Math.ceil(n / br); i++) {
            commandEncoder.copyBufferToBuffer(matrixQBuffer, (br * i * d + 2) * Float32Array.BYTES_PER_ELEMENT,
                                              patchQBuffer, 2 * Float32Array.BYTES_PER_ELEMENT,
                                              Math.min(br, n - i * br) * d * Float32Array.BYTES_PER_ELEMENT);
            commandEncoder.copyBufferToBuffer(matrixOBuffer, (br * i * d + 2) * Float32Array.BYTES_PER_ELEMENT,
                                              patchOBuffer, 2 * Float32Array.BYTES_PER_ELEMENT,
                                              Math.min(br, n - i * br) * d * Float32Array.BYTES_PER_ELEMENT);
            commandEncoder.copyBufferToBuffer(matrixLBuffer, (br * i + 2) * Float32Array.BYTES_PER_ELEMENT,
                                              patchLBuffer, 2 * Float32Array.BYTES_PER_ELEMENT,
                                              Math.min(br, n - i * br) * Float32Array.BYTES_PER_ELEMENT);
            commandEncoder.copyBufferToBuffer(matrixMBuffer, (br * i + 2) * Float32Array.BYTES_PER_ELEMENT,
                                              patchMBuffer, 2 * Float32Array.BYTES_PER_ELEMENT,
                                              Math.min(br, n - i * br) * Float32Array.BYTES_PER_ELEMENT);
            const passEncoder1 = commandEncoder.beginComputePass();
            passEncoder1.setPipeline(pipeline1);
            passEncoder1.setBindGroup(0, bindGroup1);
            passEncoder1.dispatchWorkgroups(Math.ceil(br / 8), Math.ceil(bc / 8));
            passEncoder1.end();
            const passEncoder2 = commandEncoder.beginComputePass();
            passEncoder2.setPipeline(pipeline2);
            passEncoder2.setBindGroup(0, bindGroup2);
            passEncoder2.dispatchWorkgroups(Math.ceil(br / 64));
            passEncoder2.end();
            const passEncoder3 = commandEncoder.beginComputePass();
            passEncoder3.setPipeline(pipeline3);
            passEncoder3.setBindGroup(0, bindGroup3);
            passEncoder3.dispatchWorkgroups(Math.ceil(br / 8), Math.ceil(d / 8));
            passEncoder3.end();
            commandEncoder.copyBufferToBuffer(patchOBuffer, 2 * Float32Array.BYTES_PER_ELEMENT,
                                              matrixOBuffer, (br * i * d + 2) * Float32Array.BYTES_PER_ELEMENT,
                                              Math.min(br, n - i * br) * d * Float32Array.BYTES_PER_ELEMENT);
            commandEncoder.copyBufferToBuffer(newLBuffer, 2 * Float32Array.BYTES_PER_ELEMENT,
                                              matrixLBuffer, (br * i + 2) * Float32Array.BYTES_PER_ELEMENT,
                                              Math.min(br, n - i * br) * Float32Array.BYTES_PER_ELEMENT);
            commandEncoder.copyBufferToBuffer(newMBuffer, 2 * Float32Array.BYTES_PER_ELEMENT,
                                              matrixMBuffer, (br * i + 2) * Float32Array.BYTES_PER_ELEMENT,
                                              Math.min(br, n - i * br) * Float32Array.BYTES_PER_ELEMENT);
        }
    }
    // Output
    commandEncoder.copyBufferToBuffer(matrixOBuffer, 0, stagingOBuffer, 0, (n * d + 2) * Float32Array.BYTES_PER_ELEMENT);
    const commands = commandEncoder.finish();
    device.queue.submit([commands]);

    // === Read the result ===
    await stagingOBuffer.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(stagingOBuffer.getMappedRange());
    result[0] = n; result[1] = d;
    const copiedResult = Array.from(result);
    stagingOBuffer.unmap();

    // === Display the result ===
    const inputStringQ = matrixToString(matrixQ);
    const inputStringK = matrixToString(matrixK);
    const inputStringV = matrixToString(matrixV);
    const resultString = matrixToString(copiedResult);
    document.getElementById("result").innerHTML = "<table><tr><td>Input Q:\n" + inputStringQ + "</td>" + 
                                                  "<td>Input K:\n" + inputStringK + "</td>" +
                                                  "<td>Input V:\n" + inputStringV + "</td></tr></table>" +
                                                  "Attention Result:\n" + resultString;
}
