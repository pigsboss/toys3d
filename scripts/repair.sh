#!/bin/bash

# 1. Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -i|--input-mesh) input_mesh="$2"; shift ;;
        -o|--output-mesh) output_mesh="$2"; shift ;;
        -d|--neighborhood-depth) neighborhood_depth="$2"; shift ;;
        -h|--help)
            echo "Usage: $0 -i <input_mesh> -o <output_mesh> -d <neighborhood_depth>"
            exit 0
            ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# 2. Validate that all required variables are set
if [[ -z "${input_mesh}" || -z "${output_mesh}" || -z "${neighborhood_depth}" ]]; then
    echo "Error: Missing required arguments."
    echo "Usage: $0 -i <input_mesh> -o <output_mesh> -d <neighborhood_depth>"
    exit 1
fi

# 3. Execution logic
rm -rf extracted_components hole_diagnosis_report repaired_components
python src/toys3d/meshinspect.py "${input_mesh}" --hole-diagnosis
# Note: Replaced hardcoded 'epson_seifert_applied.stl' with variable "${input_mesh}"
python src/toys3d/extract_components.py "${input_mesh}" --boundary-type healthy --boundary-neighborhood-depth "${neighborhood_depth}" --overwrite
find extracted_components -name "healthy_*_depth${neighborhood_depth}.ply" -print0 | parallel -0 -j 16 python src/toys3d/meshrepair.py {} repaired_components/{/.}_repaired.ply --status-json repaired_components/{/.}_repaired.status.json
python src/toys3d/repair_report.py --output-dir repaired_components --summary-json repaired_components/repair_summary.json --report-csv repaired_components/repair_report.csv --failed-features-json repaired_components/failed_features.json
python src/toys3d/repair_apply.py "${input_mesh}" "${output_mesh}" --output-dir repaired_components --delete-components