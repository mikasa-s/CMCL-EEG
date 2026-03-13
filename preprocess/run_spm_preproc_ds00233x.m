function run_spm_preproc_ds00233x(subject_ids, task_names, dataset_root, parallel_workers)

% SPM12 preprocessing for XP-style datasets (ds002336 / ds002338).
% Steps:
% 1) discard initial lead-in volumes
% 2) slice timing correction using BIDS JSON SliceTiming
% 3) spatial realignment
% 4) coregister T1 to mean EPI
% 5) segment T1 and normalize to MNI
% 6) smooth normalized fMRI with a 6 mm Gaussian kernel
% Only the final model-ready fMRI file is retained under each subject folder.
%
% Recommended VS Code / MATLAB setup before running:
%   addpath('D:\202406\spm\spm12');
%   savepath;
%   spm('Defaults', 'fMRI');
%   spm_jobman('initcfg');
%
% Usage:
%   run_spm_preproc_ds00233x()
%   run_spm_preproc_ds00233x('sub-xp101', 'task-eegfmriNF')
%   run_spm_preproc_ds00233x({'sub-xp101','sub-xp102'}, {'task-eegNF','task-fmriNF'}, 'D:\OpenNeuro\ds002338', 4)

    if nargin < 4 || isempty(parallel_workers)
        parallel_workers = 1;
    end

    if nargin < 3 || isempty(dataset_root)
        root_dir = 'D:\OpenNeuro\ds002336';
    else
        root_dir = char(dataset_root);
    end
    output_root = fullfile(root_dir, 'derivatives', 'spm12_preproc');

    % Dataset README says the scanner started 2 s before protocol onset.
    % With TR = 2 s this corresponds to discarding the first 1 volume.
    n_discard = 1;
    smooth_fwhm = [6 6 6];
    normalize_vox = [2 2 2];
    normalize_bb = [-78 -112 -70; 78 76 85];
    overwrite_existing = true;

    if nargin < 1 || isempty(subject_ids)
        subject_ids = list_subjects(root_dir);
    else
        subject_ids = normalize_cellstr(subject_ids);
    end

    if nargin < 2 || isempty(task_names)
        task_names = {};
    else
        task_names = normalize_cellstr(task_names);
    end

    if ~exist(output_root, 'dir')
        mkdir(output_root);
    end

    if exist('spm', 'file') ~= 2
        default_spm_dir = 'D:\202406\spm\spm12';
        if exist(default_spm_dir, 'dir') ~= 7
            error('SPM12 is not on the MATLAB path, and the default directory does not exist: %s', default_spm_dir);
        end
        addpath(default_spm_dir);
    end

    spm('Defaults', 'fMRI');
    spm_jobman('initcfg');

    has_parallel = (exist('gcp', 'file') == 2) && (exist('parpool', 'file') == 2);

    if parallel_workers > 1 && numel(subject_ids) > 1 && has_parallel
        pool = gcp('nocreate');
        if isempty(pool)
            parpool(parallel_workers);
        elseif pool.NumWorkers ~= parallel_workers
            delete(pool);
            parpool(parallel_workers);
        end

        parfor subject_index = 1:numel(subject_ids)
            process_subject_spm(root_dir, output_root, subject_ids{subject_index}, task_names, n_discard, smooth_fwhm, normalize_vox, normalize_bb, overwrite_existing);
        end
    else
        if parallel_workers > 1 && ~has_parallel
            fprintf('Parallel Computing Toolbox is not available. Falling back to serial execution.\n');
        end
        for subject_index = 1:numel(subject_ids)
            process_subject_spm(root_dir, output_root, subject_ids{subject_index}, task_names, n_discard, smooth_fwhm, normalize_vox, normalize_bb, overwrite_existing);
        end
    end

    fprintf('\nAll requested preprocessing jobs finished.\n');
    fprintf('Outputs are under: %s\n', output_root);
end


function out_file = find_one_file(folder, pattern1, pattern2)
    d = dir(fullfile(folder, pattern1));
    if isempty(d) && ~isempty(pattern2)
        d = dir(fullfile(folder, pattern2));
    end
    if isempty(d)
        error('No file found in %s matching %s or %s', folder, pattern1, pattern2);
    end
    if numel(d) > 1
        fprintf('Multiple files found. Using first one:\n');
        for i = 1:numel(d)
            fprintf('  %s\n', d(i).name);
        end
    end
    out_file = fullfile(folder, d(1).name);
end


function out_file = stage_nifti(in_file, out_dir, overwrite_existing)
    [~, n, ext] = fileparts(in_file);
    if strcmp(ext, '.gz')
        out_file = fullfile(out_dir, n);
        if isfile(out_file) && (overwrite_existing || ~is_valid_nifti(out_file))
            delete(out_file);
        end
        if overwrite_existing || ~isfile(out_file)
            gunzip(in_file, out_dir);
        end
    else
        out_file = fullfile(out_dir, [n ext]);
        if isfile(out_file) && (overwrite_existing || ~is_valid_nifti(out_file))
            delete(out_file);
        end
        if overwrite_existing || ~isfile(out_file)
            copyfile(in_file, out_file);
        end
    end
end


function out_file = discard_initial_volumes(in_file, n_discard, overwrite_existing)
    [p, n, ext] = fileparts(in_file);
    if strcmp(ext, '.nii')
        out_file = fullfile(p, ['trim_' n ext]);
    else
        out_file = fullfile(p, ['trim_' n '.nii']);
    end
    if isfile(out_file) && (overwrite_existing || ~is_valid_nifti(out_file))
        delete(out_file);
    end
    if ~overwrite_existing && isfile(out_file)
        return;
    end

    V = spm_vol(in_file);
    if numel(V) <= n_discard
        error('Not enough volumes in %s to discard %d', in_file, n_discard);
    end

    Y = spm_read_vols(V);
    Y = Y(:,:,:,n_discard+1:end);

    Vout = V(n_discard+1:end);
    for i = 1:numel(Vout)
        Vout(i).fname = out_file;
        Vout(i).n = [i 1];
    end

    spm_write_vol_4d(Vout, Y);
end


function out_scans = add_prefix_to_scans(in_scans, prefix)
    out_scans = cell(size(in_scans));
    for i = 1:numel(in_scans)
        entry = in_scans{i};
        parts = strsplit(entry, ',');
        fname = parts{1};
        idx = parts{2};

        [p, n, ext] = fileparts(fname);
        new_fname = fullfile(p, [prefix n ext]);
        out_scans{i} = sprintf('%s,%s', new_fname, idx);
    end
end


function spm_write_vol_4d(V, Y)
    for i = 1:numel(V)
        spm_write_vol(V(i), Y(:,:,:,i));
    end
end


function scans = build_scan_list(nifti_file)
    V = spm_vol(nifti_file);
    scans = cell(numel(V), 1);
    for i = 1:numel(V)
        scans{i} = sprintf('%s,%d', nifti_file, i);
    end
end


function out_path = mean_image_path(nifti_file, prefix)
    [p, n, ~] = fileparts(nifti_file);
    out_path = fullfile(p, ['mean' prefix n '.nii']);
end


function values = normalize_cellstr(values)
    if ischar(values)
        values = {values};
        return;
    end
    if isstring(values)
        values = cellstr(values(:));
        return;
    end
    if iscell(values)
        values = cellfun(@char, values, 'UniformOutput', false);
        return;
    end
    error('Unsupported input type for subject/task selection.');
end


function subject_ids = list_subjects(root_dir)
    d = dir(fullfile(root_dir, 'sub-*'));
    d = d([d.isdir]);
    subject_ids = sort({d.name});
end


function task_names = resolve_subject_tasks(root_dir, subject_id, requested_tasks)
    func_dir = fullfile(root_dir, subject_id, 'func');
    d = dir(fullfile(func_dir, sprintf('%s_task-*_bold.json', subject_id)));
    detected = {};
    for i = 1:numel(d)
        token = regexp(d(i).name, '_task-[^_]+(?:_run-[^_]+)?', 'match', 'once');
        if ~isempty(token)
            detected{end+1} = token(2:end); %#ok<AGROW>
        end
    end
    detected = unique(detected, 'stable');
    if isempty(requested_tasks)
        task_names = detected;
    else
        task_names = {};
        for i = 1:numel(detected)
            candidate = detected{i};
            for j = 1:numel(requested_tasks)
                requested = requested_tasks{j};
                if strcmp(candidate, requested) || startsWith(candidate, [requested '_run-'])
                    task_names{end+1} = candidate; %#ok<AGROW>
                    break;
                end
            end
        end
    end
end


function meta = read_json_file(json_path)
    raw_text = fileread(json_path);
    meta = jsondecode(raw_text);
end


function value = get_required_scalar(meta, field_name)
    if ~isfield(meta, field_name)
        error('Missing required JSON field %s.', field_name);
    end
    value = double(meta.(field_name));
end


function value = get_required_vector(meta, field_name)
    if ~isfield(meta, field_name)
        error('Missing required JSON field %s.', field_name);
    end
    value = double(meta.(field_name));
    value = value(:)';
end


function run_stage_one_batch(scans, a_scans, mean_epi, staged_anat, TR, acquisition_time, num_slices, slice_timing, ref_slice_timing, work_dir, subject_id, task_name)
    matlabbatch = {};

    matlabbatch{1}.spm.temporal.st.scans = {scans};
    matlabbatch{1}.spm.temporal.st.nslices = num_slices;
    matlabbatch{1}.spm.temporal.st.tr = TR;
    matlabbatch{1}.spm.temporal.st.ta = acquisition_time;
    matlabbatch{1}.spm.temporal.st.so = slice_timing;
    matlabbatch{1}.spm.temporal.st.refslice = ref_slice_timing;
    matlabbatch{1}.spm.temporal.st.prefix = 'a';

    matlabbatch{2}.spm.spatial.realign.estwrite.data = {a_scans};
    matlabbatch{2}.spm.spatial.realign.estwrite.eoptions.quality = 0.9;
    matlabbatch{2}.spm.spatial.realign.estwrite.eoptions.sep = 4;
    matlabbatch{2}.spm.spatial.realign.estwrite.eoptions.fwhm = 5;
    matlabbatch{2}.spm.spatial.realign.estwrite.eoptions.rtm = 1;
    matlabbatch{2}.spm.spatial.realign.estwrite.eoptions.interp = 2;
    matlabbatch{2}.spm.spatial.realign.estwrite.eoptions.wrap = [0 0 0];
    matlabbatch{2}.spm.spatial.realign.estwrite.eoptions.weight = '';
    matlabbatch{2}.spm.spatial.realign.estwrite.roptions.which = [2 1];
    matlabbatch{2}.spm.spatial.realign.estwrite.roptions.interp = 4;
    matlabbatch{2}.spm.spatial.realign.estwrite.roptions.wrap = [0 0 0];
    matlabbatch{2}.spm.spatial.realign.estwrite.roptions.mask = 1;
    matlabbatch{2}.spm.spatial.realign.estwrite.roptions.prefix = 'r';

    matlabbatch{3}.spm.spatial.coreg.estimate.ref = {mean_epi};
    matlabbatch{3}.spm.spatial.coreg.estimate.source = {staged_anat};
    matlabbatch{3}.spm.spatial.coreg.estimate.other = {''};
    matlabbatch{3}.spm.spatial.coreg.estimate.eoptions.cost_fun = 'nmi';
    matlabbatch{3}.spm.spatial.coreg.estimate.eoptions.sep = [4 2];
    matlabbatch{3}.spm.spatial.coreg.estimate.eoptions.tol = [0.0200 0.0200 0.0200 0.0010 0.0010 0.0010 0.0100 0.0100 0.0100 0.0010 0.0010 0.0010];
    matlabbatch{3}.spm.spatial.coreg.estimate.eoptions.fwhm = [7 7];

    tpm_path = fullfile(spm('Dir'), 'tpm', 'TPM.nii');
    matlabbatch{4}.spm.spatial.preproc.channel.vols = {staged_anat};
    matlabbatch{4}.spm.spatial.preproc.channel.biasreg = 0.001;
    matlabbatch{4}.spm.spatial.preproc.channel.biasfwhm = 60;
    matlabbatch{4}.spm.spatial.preproc.channel.write = [0 1];

    for tissue_index = 1:6
        matlabbatch{4}.spm.spatial.preproc.tissue(tissue_index).tpm = {sprintf('%s,%d', tpm_path, tissue_index)};
        if tissue_index == 1
            ngaus = 1;
        elseif tissue_index == 2
            ngaus = 1;
        elseif tissue_index == 3
            ngaus = 2;
        elseif tissue_index == 4
            ngaus = 3;
        elseif tissue_index == 5
            ngaus = 4;
        else
            ngaus = 2;
        end
        matlabbatch{4}.spm.spatial.preproc.tissue(tissue_index).ngaus = ngaus;
        matlabbatch{4}.spm.spatial.preproc.tissue(tissue_index).native = [0 0];
        matlabbatch{4}.spm.spatial.preproc.tissue(tissue_index).warped = [0 0];
    end

    matlabbatch{4}.spm.spatial.preproc.warp.mrf = 1;
    matlabbatch{4}.spm.spatial.preproc.warp.cleanup = 1;
    matlabbatch{4}.spm.spatial.preproc.warp.reg = [0 0.001 0.5 0.05 0.2];
    matlabbatch{4}.spm.spatial.preproc.warp.affreg = 'mni';
    matlabbatch{4}.spm.spatial.preproc.warp.fwhm = 0;
    matlabbatch{4}.spm.spatial.preproc.warp.samp = 3;
    matlabbatch{4}.spm.spatial.preproc.warp.write = [1 1];

    save(fullfile(work_dir, sprintf('%s_%s_spm_stage1.mat', subject_id, task_name)), 'matlabbatch');
    spm_jobman('run', matlabbatch);
end


function run_stage_two_batch(deformation_field, staged_anat, ra_scans, wa_ra_scans, smooth_fwhm, normalize_vox, normalize_bb, work_dir, subject_id, task_name)
    matlabbatch = {};

    matlabbatch{1}.spm.spatial.normalise.write.subj.def = {deformation_field};
    matlabbatch{1}.spm.spatial.normalise.write.subj.resample = [{staged_anat}; ra_scans];
    matlabbatch{1}.spm.spatial.normalise.write.woptions.bb = normalize_bb;
    matlabbatch{1}.spm.spatial.normalise.write.woptions.vox = normalize_vox;
    matlabbatch{1}.spm.spatial.normalise.write.woptions.interp = 4;
    matlabbatch{1}.spm.spatial.normalise.write.woptions.prefix = 'w';

    matlabbatch{2}.spm.spatial.smooth.data = wa_ra_scans;
    matlabbatch{2}.spm.spatial.smooth.fwhm = smooth_fwhm;
    matlabbatch{2}.spm.spatial.smooth.dtype = 0;
    matlabbatch{2}.spm.spatial.smooth.im = 0;
    matlabbatch{2}.spm.spatial.smooth.prefix = 's';

    save(fullfile(work_dir, sprintf('%s_%s_spm_stage2.mat', subject_id, task_name)), 'matlabbatch');
    spm_jobman('run', matlabbatch);
end


function final_fmri = finalize_task_outputs(subject_out_dir, task_work_dir, subject_id, task_name, overwrite_existing)
    final_fmri = fullfile(subject_out_dir, sprintf('%s_%s_bold.nii', subject_id, task_name));
    final_candidate = fullfile(task_work_dir, sprintf('swratrim_%s_%s_bold.nii', subject_id, task_name));

    if ~isfile(final_candidate)
        error('Final smoothed SPM output not found: %s', final_candidate);
    end

    if overwrite_existing || ~isfile(final_fmri)
        if isfile(final_fmri)
            delete(final_fmri);
        end
        movefile(final_candidate, final_fmri);
    else
        delete(final_candidate);
    end

    cleanup_task_directory(task_work_dir);
end


function cleanup_task_directory(task_work_dir)
    if exist(task_work_dir, 'dir') ~= 7
        return;
    end
    rmdir(task_work_dir, 's');
end


function ok = is_valid_nifti(nifti_path)
    ok = false;
    if ~isfile(nifti_path)
        return;
    end
    try
        V = spm_vol(nifti_path);
        ok = ~isempty(V);
    catch
        ok = false;
    end
end


function process_subject_spm(root_dir, output_root, subject_id, task_names, n_discard, smooth_fwhm, normalize_vox, normalize_bb, overwrite_existing)
    subject_tasks = resolve_subject_tasks(root_dir, subject_id, task_names);
    if isempty(subject_tasks)
        fprintf('No matching tasks found for %s. Skipping.\n', subject_id);
        return;
    end

    anat_dir = fullfile(root_dir, subject_id, 'anat');
    anat_file = find_one_file(anat_dir, sprintf('%s*_T1w.nii', subject_id), sprintf('%s*_T1w.nii.gz', subject_id));

    for task_index = 1:numel(subject_tasks)
        task_name = subject_tasks{task_index};
        fprintf('\n[%d/%d] %s %s\n', task_index, numel(subject_tasks), subject_id, task_name);

        func_dir = fullfile(root_dir, subject_id, 'func');
        func_file = find_one_file(func_dir, sprintf('%s_%s*_bold.nii', subject_id, task_name), sprintf('%s_%s*_bold.nii.gz', subject_id, task_name));
        func_json = find_one_file(func_dir, sprintf('%s_%s*_bold.json', subject_id, task_name), '');

        meta = read_json_file(func_json);
        TR = get_required_scalar(meta, 'RepetitionTime');
        slice_timing = get_required_vector(meta, 'SliceTiming');
        num_slices = numel(slice_timing);
        acquisition_time = max(slice_timing) - min(slice_timing);
        nominal_ref_timing = median(slice_timing);
        [~, ref_slice_index] = min(abs(slice_timing - nominal_ref_timing));
        ref_slice_timing = slice_timing(ref_slice_index);

        subject_out_dir = fullfile(output_root, subject_id);
        task_work_dir = fullfile(subject_out_dir, ['_tmp_' task_name]);
        if ~exist(subject_out_dir, 'dir')
            mkdir(subject_out_dir);
        end
        if ~exist(task_work_dir, 'dir')
            mkdir(task_work_dir);
        end

        staged_func = stage_nifti(func_file, task_work_dir, overwrite_existing);
        staged_anat = stage_nifti(anat_file, task_work_dir, overwrite_existing);
        trimmed_func = discard_initial_volumes(staged_func, n_discard, overwrite_existing);

        scans = build_scan_list(trimmed_func);
        a_scans = add_prefix_to_scans(scans, 'a');
        ra_scans = add_prefix_to_scans(a_scans, 'r');
        wa_ra_scans = add_prefix_to_scans(ra_scans, 'w');
        mean_epi = mean_image_path(trimmed_func, 'a');

        [~, anat_name, anat_ext] = fileparts(staged_anat);
        deformation_field = fullfile(task_work_dir, ['y_' anat_name anat_ext]);

        fprintf('  Functional: %s\n', staged_func);
        fprintf('  Anatomical: %s\n', staged_anat);
        fprintf('  JSON TR: %.4f s\n', TR);
        fprintf('  JSON slices: %d\n', num_slices);
        fprintf('  JSON SliceTiming: %s\n', mat2str(slice_timing));
        fprintf('  JSON TA: %.4f s\n', acquisition_time);
        fprintf('  Reference slice index: %d\n', ref_slice_index);
        fprintf('  Reference slice timing: %.4f s\n', ref_slice_timing);

        run_stage_one_batch(scans, a_scans, mean_epi, staged_anat, TR, acquisition_time, num_slices, slice_timing, ref_slice_timing, task_work_dir, subject_id, task_name);

        if ~isfile(deformation_field)
            error('SPM segmentation did not produce deformation field: %s', deformation_field);
        end

        run_stage_two_batch(deformation_field, staged_anat, ra_scans, wa_ra_scans, smooth_fwhm, normalize_vox, normalize_bb, task_work_dir, subject_id, task_name);
        final_fmri = finalize_task_outputs(subject_out_dir, task_work_dir, subject_id, task_name, overwrite_existing);
        fprintf('  Final model-ready fMRI: %s\n', final_fmri);
    end
end