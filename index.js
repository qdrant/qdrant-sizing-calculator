
const datatypeSizeBytes = {
    float32: 4, // Default
    float16: 2,
    uint8: 1,
}

const quentizationSizeFactor = {
    scalar: 1,
    binary: 1/8,
    productX4: 1,
    productX8: 1/2,
    productX16: 1/4,
    productX32: 1/8,
    productX64: 1/16,
}


// Number of bytes in usize type
const uSize = 4;


export function estimateQdrantClusterFull({
    numberOfVectors,
    dimensions,
    datatype = 'float32', // Default value of qdrant
    hnswM = 16, // Default value of qdrant
    hnswOnDisk = false, // Usually it is false
    vectorsOnDisk = false,
    quantization = null, // Type of quantization
    quantizationOnDisk = false, // Usually it is false, cause `always_ram` is used
    sparseVectors = 0,
    elementsPerSparseVector = 0,
    sparseVectorDatatype = 'float32',
    sparseVectorsOnDisk = false,
    hotSubset = 0.0, // For multi-tenant use-cases. Estimate the portion of data which should actually be in RAM
}) {

    let ramSize = 0;
    let diskSize = 0;


    // Check datatype exists

    if (!datatypeSizeBytes[datatype]) {
        throw new Error(`Unknown datatype: ${datatype}, available: ${Object.keys(datatypeSizeBytes).join(', ')}`);
    }

    // Vectors
    const vectorsDiskSize = numberOfVectors * dimensions * datatypeSizeBytes[datatype];

    diskSize += vectorsDiskSize;

    // Vectors are also stored in RAM
    if (vectorsOnDisk) {
        ramSize += vectorsDiskSize * hotSubset;
    } else {
        ramSize += vectorsDiskSize;
    }

    // Quantization

    if (quantization) {
        if (!quentizationSizeFactor[quantization]) {
            throw new Error(`Unknown quantization: ${quantization}, available: ${Object.keys(quentizationSizeFactor).join(', ')}`);
        }

        const quantizationSize = numberOfVectors * dimensions * quentizationSizeFactor[quantization];

        diskSize += quantizationSize;

        if (quantizationOnDisk) {
            ramSize += quantizationSize * hotSubset;
        } else {
            ramSize += quantizationSize;
        }
    }

    // HNSW

    const actualHnswM = hnswM * 2; // Internally, largest layer of HNSW has 2 * M connections

    const hnswSize = numberOfVectors * actualHnswM * uSize;

    diskSize += hnswSize;

    if (hnswOnDisk) {
        ramSize += hnswSize * hotSubset
    } else {
        ramSize += hnswSize;
    } 

    // Sparse vectors

    if (!datatypeSizeBytes[sparseVectorDatatype]) {
        throw new Error(`Unknown datatype: ${sparseVectorDatatype}, available: ${Object.keys(datatypeSizeBytes).join(', ')}`);
    }

    const sparseVectorsSize = sparseVectors * (elementsPerSparseVector * (datatypeSizeBytes[sparseVectorDatatype] + uSize));

    diskSize += sparseVectorsSize * 2; // We store sparse vectors twice, once for index and once for data

    if (sparseVectorsOnDisk) {
        ramSize += sparseVectorsSize * hotSubset;
    } else {
        ramSize += sparseVectorsSize;
    }

    // ID tacker = 24bytes per point

    const forwardReference = uSize * 2; // 8
    const backwardReference = uSize * 2; // 8
    const versionReference = uSize * 2; // 8

    const idTrackerSize = numberOfVectors * (forwardReference + backwardReference + versionReference);

    diskSize += idTrackerSize;
    ramSize += idTrackerSize;


    return {
        ramSize,
        diskSize,
    };
}