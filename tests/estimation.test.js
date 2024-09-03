// Tests for the estimation module

import { estimateQdrantClusterFull } from '../index';


test('Estimate Qdrant cluster default', () => {
    const result = estimateQdrantClusterFull({
        numberOfVectors: 1000,
        dimensions: 100,
    });

    // Expect RAM and disk size to be equal
    expect(result.ramSize).toBe(result.diskSize);

});


test("Disk usage with quantization is bigger than without", () => {

    const resultDefault = estimateQdrantClusterFull({
        numberOfVectors: 1000,
        dimensions: 100,
    });

    const resultWithQuantization = estimateQdrantClusterFull({
        numberOfVectors: 1000,
        dimensions: 100,
        quantization: 'scalar',
    });

    expect(resultWithQuantization.diskSize).toBeGreaterThan(resultDefault.diskSize);
});


test("Reduce RAM usage by putting vectors on disk and using binary quantization", () => {

    const resultDefault = estimateQdrantClusterFull({
        numberOfVectors: 1_000_000,
        dimensions: 1024,
    });

    const resultWithVectorsOnDisk = estimateQdrantClusterFull({
        numberOfVectors: 1_000_000,
        dimensions: 1024,
        vectorsOnDisk: true,
    });

    const resultWithVectorsOnDiskAndQuantization = estimateQdrantClusterFull({
        numberOfVectors: 1_000_000,
        dimensions: 1024,
        vectorsOnDisk: true,
        quantization: 'binary',
        quantizationOnDisk: false,
    });


    expect(
        resultDefault.ramSize > resultWithVectorsOnDiskAndQuantization.ramSize &&
        resultWithVectorsOnDiskAndQuantization.ramSize > resultWithVectorsOnDisk.ramSize
    ).toBe(true);

    // Expect quatization to increase disk usage for both with and without vectors on disk

    expect(resultDefault.diskSize).toBeLessThan(resultWithVectorsOnDiskAndQuantization.diskSize);
    expect(resultWithVectorsOnDisk.diskSize).toBeLessThan(resultWithVectorsOnDiskAndQuantization.diskSize);
});


test("Smaller datatype reduces both RAM and disk usage", () => {

    const resultDefault = estimateQdrantClusterFull({
        numberOfVectors: 1_000_000,
        dimensions: 1024,
    });

    const resultWithFloat16 = estimateQdrantClusterFull({
        numberOfVectors: 1_000_000,
        dimensions: 1024,
        datatype: 'float16',
    });

    expect(resultWithFloat16.ramSize).toBeLessThan(resultDefault.ramSize);
    expect(resultWithFloat16.diskSize).toBeLessThan(resultDefault.diskSize);
});


test("Sanity check for large amount of vectors", () => {

    const resultDefault = estimateQdrantClusterFull({
        numberOfVectors: 1_000_000,
        dimensions: 1024,
        vectorsOnDisk: true,
        datatype: 'float16',
        quantization: "scalar",
        quantizationOnDisk: false,
    });

    const diskSizeGb = resultDefault.diskSize / 1024 / 1024 / 1024;
    const ramSizeGb = resultDefault.ramSize / 1024 / 1024 / 1024;



    expect(diskSizeGb).toBeGreaterThan(2);
    expect(diskSizeGb).toBeLessThan(4);

    expect(ramSizeGb).toBeGreaterThan(1);
    expect(ramSizeGb).toBeLessThan(2);
});

