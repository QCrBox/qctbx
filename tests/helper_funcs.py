def new_scif_with_workdir(input_scif_path, work_dir, output_scif_path):
    """
    Enables the use of temporary directories for testing purposes.
    Replace the placeholder $WORKDIRPLACEHOLDER in the input scif file with the
    string representation of the temp directory and write the output scif file.
    """
    work_dir_str = str(work_dir)
    with open(input_scif_path, 'r', encoding='ASCII') as fo:
        scif_content = fo.read()

    with open(output_scif_path, 'w', encoding='ASCII') as fo:
        fo.write(scif_content.replace("$WORKDIRPLACEHOLDER", work_dir_str))
