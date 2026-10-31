import json
import subprocess
import sys


def test_doctor_command_is_usable_as_an_installed_module():
    result = subprocess.run([sys.executable, '-m', 'reactoccflow', 'doctor', '--cpu-only'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report['package_version'] == '0.1.0'
    assert report['cpu_tensor_check'] is True
    assert report['cuda_available'] is None


def test_data_inspection_reports_valid_windows(tmp_path):
    import numpy as np
    import pickle
    from reactoccflow.data.cache import convert_legacy_cache
    records = [{'x_sampled': np.zeros((1,2,2,2),np.float32),
                'gt_path': [f'gts/scene-0001/t{i}/labels.npz'],
                'gt_trajs': np.zeros((1,2),np.float32)} for i in range(4)]
    path = tmp_path/'source.pkl'
    with path.open('wb') as f:
        pickle.dump(records,f)
    info = tmp_path / 'info.pkl'
    with info.open('wb') as f:
        pickle.dump({'infos': {'scene-0001': [
            {'token': f't{i}', 'timestamp': i * 500000} for i in range(4)]}}, f)
    output=tmp_path/'dataset'
    convert_legacy_cache(path,output,info_path=info)
    result = subprocess.run([sys.executable,'-m','reactoccflow','inspect-cache',str(output),'--history','2','--future','1'], capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    report=json.loads(result.stdout)
    assert report['frames']==4
    assert report['valid_windows']==2
    assert report['coordinate_frame']=='legacy_per_frame_ego'
