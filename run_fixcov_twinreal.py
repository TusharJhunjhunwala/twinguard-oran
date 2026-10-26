import fixcov, sys, json, os, time
sys.argv=['x','--study','twinreal','--jobs','2']
import twinguard_rev3 as X
X.main()
os.replace('outputs_rev3/twinreal.json','outputs_rev3/twinreal_stdcov.json')
