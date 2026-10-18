import json
from tapestry_m1.cli import main


def test_cli_evaluate_compare_and_errors(benchmark,rankings,tmp_path,capsys):
    output=tmp_path/'result.json'
    assert main(['evaluate',str(benchmark),str(rankings),'--output',str(output)])==0
    assert json.loads(output.read_text())['counts']['evaluated']==2
    capsys.readouterr()
    assert main(['compare',str(output),str(output),'--resamples','100'])==0
    comparison=json.loads(capsys.readouterr().out)
    assert comparison['groups'][0]['metrics']['mrr']['ci95']==[0,0]
    assert main(['evaluate',str(benchmark),str(rankings),'--mrr-depth','0'])==2
    assert json.loads(capsys.readouterr().err)['status']=='error'
