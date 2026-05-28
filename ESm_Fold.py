conda create -n esmfold python=3.10 -y
conda activate esmfold
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install fair-esm
pip install openfold
pip install biotite py3Dmol jupyterlab
import torch
import esm

device = "cuda" if torch.cuda.is_available() else "cpu"

model = esm.pretrained.esmfold_v1()
model = model.eval().to(device)

# Reduce memory spikes
model.set_chunk_size(128)

print("Model loaded on:", device)
sequence = "MKTFFVAGLFLALALAGALAAPVSA"

with torch.no_grad():
    pdb_output = model.infer_pdb(sequence)

print("Prediction complete!")
with open("predicted_structure.pdb", "w") as f:
    f.write(pdb_output)

print("PDB saved!")
import py3Dmol

view = py3Dmol.view(width=800, height=600)
view.addModel(pdb_output, 'pdb')
view.setStyle({'cartoon': {'color': 'spectrum'}})
view.zoomTo()
view.show()
