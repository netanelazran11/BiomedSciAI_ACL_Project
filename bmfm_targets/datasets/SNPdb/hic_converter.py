import argparse
import os
from pathlib import Path

import hicstraw
import numpy as np
import pandas as pd
from scipy import sparse

from bmfm_targets.datasets.SNPdb.tabix_converter import (
    extract_chr_seq_and_len,
    sample_variant,
)


def sequential_sample_hic_contact_chunks(
    sequence: str,
    hic_path: str | Path,
    output_path: str | Path,
    target_chr="chr22",
    hic_resolution=1000,
    number_of_chunks=10,
):
    """
    Read a Hi-C file and DNA sequence, sequentially create samples of multiple DNA chunks separated by "#", and their pairwise contact scores from Hi-C.

    Args:
    ----
        sequence (str): DNA sequence (variation-encoded or reference) of the target chromosome
        hic_path (str): input Hi-C file
        output_path (str): output file
        target_chr (str): the target chromosome to be processed
        hic_resolution (int): the basic unit of Hi-C contact
        number_of_chunks (int): the number of chunks within a sample

    Returns:
    -------
        Number of samples and the total number of contacts
        (the sample of DNA chunks and pairwise contact scores will be written in the output file).
    """
    hic = hicstraw.HiCFile(hic_path)
    mzd = hic.getMatrixZoomData(
        target_chr[3:], target_chr[3:], "oe", "KR", "BP", hic_resolution
    )
    sequence_len = len(sequence)
    sample_len = hic_resolution * number_of_chunks
    number_of_samples = 0
    number_of_contacts = 0
    with open(output_path, "w") as f:
        for i in range(0, sequence_len // sample_len):
            left = i * sample_len
            right = left + hic_resolution * (number_of_chunks - 1)
            contact_matrix = mzd.getRecordsAsMatrix(left, right, left, right)
            contact_count = np.sum(np.triu(contact_matrix, k=1) != 0)
            ## if any inter-chunk contact exists, create a sample of multiple chunks
            if contact_count > 0:
                label = contact_matrix[np.triu_indices(number_of_chunks, k=1)]
                dna_chunks = [
                    sequence[x : (x + hic_resolution)]
                    for x in range(left, right + hic_resolution, hic_resolution)
                ]
                # here I use "#" as [SEP] for now
                sample_sequence = "#".join(dna_chunks)
                f.write(
                    ",".join(dna_chunks + label.round(2).astype("str").tolist()) + "\n"
                )
                number_of_samples += 1
                number_of_contacts += contact_count
    return number_of_samples, number_of_contacts


def random_sample_hic_contact_chunks(
    sequence: str,
    hic_path: str | Path,
    output_path: str | Path,
    target_chr="chr22",
    hic_resolution=1000,
    number_of_chunks=10,
    shifted_bin=0,
    sample_ratio=1.0,
    seed=0,
    exclude_N_islands=True,
    binarize_label=True,
):
    """
    Read a Hi-C file and DNA sequence, randomly create samples of multiple DNA chunks separated by "#", and their pairwise contact scores from Hi-C.

    Args:
    ----
        sequence (str): DNA sequence (variation-encoded or reference) of the target chromosome
        hic_path (str): input Hi-C file
        output_path (str): output file
        target_chr (str): the target chromosome to be processed
        hic_resolution (int): the basic unit of Hi-C contact
        number_of_chunks (int): the number of chunks within a sample
        shifted_bin (int): the shifted bins if not start with 0
        sample_ratio (float): the sample_ratio of the chromosome/TAD - 1.0 means the total length of samples is roughly the same length of the chromosome/TAD. Higher means more samples
        exclude_N_islands (boolean): if True, all-N bins will be excluded from sampling
        binarize_label (boolean): if True, convert label into 0/1 as a classification problem

    Returns:
    -------
        Number of samples and the total number of contacts
        (the sample of DNA chunks and pairwise contact scores will be written in the output file).
    """
    hic = hicstraw.HiCFile(hic_path)
    mzd = hic.getMatrixZoomData(
        target_chr[3:], target_chr[3:], "oe", "KR", "BP", hic_resolution
    )
    sequence_len = len(sequence)
    maximum_position = sequence_len // hic_resolution
    if exclude_N_islands:
        all_dna_chunks = [
            sequence[(x * hic_resolution) : (x * hic_resolution + hic_resolution)]
            for x in range(maximum_position)
        ]
        all_bins = np.where([x != {"N"} for x in [set(x) for x in all_dna_chunks]])[0]
    else:
        all_bins = range(maximum_position)
    number_of_samples = int(
        sequence_len // (hic_resolution * number_of_chunks) * sample_ratio
    )
    np.random.seed(seed)
    number_of_contacts = 0
    with open(output_path, "a") as f:
        for _ in range(0, number_of_samples):
            label = []
            bins = np.random.choice(all_bins, number_of_chunks, replace=False)
            for i in range(number_of_chunks - 1):
                for j in range(i + 1, number_of_chunks):
                    ## shift bins for hic data
                    left = (bins[i] + shifted_bin) * hic_resolution
                    right = (bins[j] + shifted_bin) * hic_resolution
                    label.append(mzd.getRecordsAsMatrix(left, left, right, right)[0, 0])
            label = np.array(label)
            dna_chunks = [
                sequence[(x * hic_resolution) : (x * hic_resolution + hic_resolution)]
                for x in bins.tolist()
            ]
            if binarize_label:
                f.write(
                    ",".join(
                        dna_chunks + (label != 0).astype("int").astype("str").tolist()
                    )
                    + "\n"
                )
            else:
                f.write(
                    ",".join(dna_chunks + label.round(2).astype("str").tolist()) + "\n"
                )
            number_of_contacts += sum(label != 0)
    return number_of_samples, number_of_contacts


def cut_sequence_by_tad_bin(sequence, target_chr, tad_path, hic_resolution=1000):
    df_tad = pd.read_csv(tad_path, header=None)
    df_tad.columns = ["chromosome", "start", "end"]
    df_tad = df_tad[df_tad.chromosome == target_chr]
    df_tad = df_tad.sort_values(by="start")
    start_all = df_tad.start.tolist()
    end_all = df_tad.end.tolist()
    ## start and end position at the hic bin level
    start_all = [x // hic_resolution * hic_resolution for x in start_all]
    end_all = [x // hic_resolution * hic_resolution for x in end_all]
    sub_sequences = []
    for start, end in zip(start_all, end_all):
        sub_sequences.append(sequence[start:end])
    return sub_sequences, start_all


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-chr",
        "--chromosome",
        help="The chromosome to be processed",
        type=str,
        required=False,
        default="chr21",
    )
    parser.add_argument(
        "--hic_path",
        help="The Hi-C file",
        type=str,
        required=False,
        default="/dccstor/bmfm-targets/data/omics/genome/hic/4DNFICSTCJQZ.hic",
    )
    parser.add_argument(
        "--hic_resolution",
        help="The resolution for Hi-C data",
        type=int,
        required=False,
        default=1000,
    )
    parser.add_argument(
        "--number_of_chunks",
        help="The number of chunks within a sample",
        type=int,
        required=False,
        default=10,
    )
    parser.add_argument(
        "--sample_strategy",
        help="sample DNA chunks sequentially or randomly",
        type=str,
        required=False,
        default="sequential",
    )
    parser.add_argument(
        "--process_ref_genome",
        help="create samples for the reference genome",
        action=argparse.BooleanOptionalAction,
        required=False,
        default=False,
    )
    parser.add_argument(
        "--output_variation_biallele_path",
        help="output directory for biallele-encoded samples",
        type=str,
        required=False,
        default="/dccstor/bmfm-targets/data/omics/genome/snpdb/raw/pretraining_data/biallele_hic_hepg2_1kb_10/",
    )
    parser.add_argument(
        "--output_reference_genome_path",
        help="output directory for reference genome samples",
        type=str,
        required=False,
        default="/dccstor/bmfm-targets/data/omics/genome/snpdb/raw/pretraining_data/ref_hic_hepg2_1kb_10/",
    )
    parser.add_argument(
        "--tad_path",
        help="input TAD file",
        type=str,
        required=False,
        default=None,
    )
    args = parser.parse_args()
    return args


def main(args):
    SNPDB_RESOURCES_PATH = (
        "/dccstor/bmfm-targets/data/omics/genome/snpdb/raw/resources/"
    )
    fasta_path = (
        SNPDB_RESOURCES_PATH + "GCA_000001405.15_GRCh38_no_alt_analysis_set.fna"
    )
    variation_matrix_path = (
        "/dccstor/bmfm-targets/data/omics/genome/snpdb/raw/matrix_snp_probability/"
    )
    chromosome = args.chromosome
    hic_resolution = args.hic_resolution
    number_of_chunks = args.number_of_chunks
    assert args.sample_strategy in ["sequential", "random"]
    if args.sample_strategy == "sequential":
        sample_function = sequential_sample_hic_contact_chunks
    elif args.sample_strategy == "random":
        sample_function = random_sample_hic_contact_chunks
    # load the variation sparse matrix
    snp_probability_matrix = sparse.load_npz(
        variation_matrix_path + "snp_prob_" + chromosome + ".npz"
    )
    # sample and encode variants in the biallele fashion
    encoded_seq = sample_variant(
        snp_probability_matrix,
        replacement=False,
    )
    ## 1. biallele encoded genome
    os.makedirs(args.output_variation_biallele_path, exist_ok=True)
    # cut into TAD sub sequences if TAD file is provided
    if args.tad_path:
        sub_seqs, starts = cut_sequence_by_tad_bin(
            encoded_seq,
            target_chr=chromosome,
            tad_path=args.tad_path,
            hic_resolution=hic_resolution,
        )

    else:
        sub_seqs = [encoded_seq]
        starts = [0]
    total_number_of_samples = 0
    total_number_of_contacts = 0
    for sequence, start in zip(sub_seqs, starts):
        if len(sequence) < hic_resolution * number_of_chunks:
            print(
                "Exclulde a short TAD of %d bp starts at %d " % (len(sequence), start)
            )
        else:
            number_of_samples, number_of_contacts = sample_function(
                sequence=sequence,
                hic_path=args.hic_path,
                output_path=args.output_variation_biallele_path
                + "sample_"
                + chromosome
                + ".txt",
                target_chr=chromosome,
                shifted_bin=start // hic_resolution,
                hic_resolution=hic_resolution,
                number_of_chunks=number_of_chunks,
                exclude_N_islands=True,
            )
            total_number_of_samples += number_of_samples
            total_number_of_contacts += number_of_contacts
    print(
        "%s has %d samples, each sample has %.1f contacts (%.1f%%) on average"
        % (
            chromosome,
            total_number_of_samples,
            total_number_of_contacts / total_number_of_samples,
            total_number_of_contacts
            / total_number_of_samples
            / number_of_chunks
            / (number_of_chunks - 1)
            * 2
            * 100,
        )
    )
    ## 2. reference genome
    if args.process_ref_genome:
        os.makedirs(args.output_reference_genome_path, exist_ok=True)
        chr_to_seq, _ = extract_chr_seq_and_len(fasta_path, ">")
        # cut into TAD sub sequences if TAD file is provided
        if args.tad_path:
            sub_seqs, starts = cut_sequence_by_tad_bin(
                chr_to_seq[chromosome],
                target_chr=chromosome,
                tad_path=args.tad_path,
                hic_resolution=hic_resolution,
            )
        else:
            sub_seqs = [chr_to_seq[chromosome]]
            starts = [0]
        for sequence, start in zip(sub_seqs, starts):
            if len(sequence) < hic_resolution * number_of_chunks:
                print(
                    "Exclulde a short TAD of %d bp starts at %d "
                    % (len(sequence), start)
                )
            else:
                number_of_samples, number_of_contacts = sample_function(
                    sequence=sequence,
                    hic_path=args.hic_path,
                    output_path=args.output_reference_genome_path
                    + "sample_"
                    + chromosome
                    + ".txt",
                    target_chr=chromosome,
                    shifted_bin=start // hic_resolution,
                    hic_resolution=hic_resolution,
                    number_of_chunks=number_of_chunks,
                    exclude_N_islands=True,
                )


if __name__ == "__main__":
    args = get_args()
    main(args)
