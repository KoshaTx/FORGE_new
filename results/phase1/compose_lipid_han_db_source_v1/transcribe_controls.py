"""Primary-drawing transcriptions for the next program; no training or replay admission."""
import json
from collections import Counter
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from forge.assembly.repeated_components import element_inventory
from forge.corpus.compose_lipid_source_view import dump, pin

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
SOURCE=ROOT/'results/phase1/compose_lipid_all_family_sources_v1/remaining/38409275'


def main():
    controls=[]
    for label,epoxide,acyl,intermediate,product,calculated,observed in [
        ('1-6-6','CCCCC1CO1','CCCCCC(=O)Cl',
         'CN(C)CCCN(CC(O)CCCC)CC(O)CCCC',
         'CN(C)CCCN(CC(OC(=O)CCCCC)CCCC)CC(OC(=O)CCCCC)CCCC',498.44,499.50),
        ('1-10-8','CCCCCCCCC1CO1','CCCCCCCC(=O)Cl',
         'CN(C)CCCN(CC(O)CCCCCCCC)CC(O)CCCCCCCC',
         'CN(C)CCCN(CC(OC(=O)CCCCCCC)CCCCCCCC)CC(OC(=O)CCCCCCC)CCCCCCCC',666.63,667.80),
    ]:
        mol=Chem.MolFromSmiles(product)
        mass=rdMolDescriptors.CalcExactMolWt(mol)
        assert round(mass,2)==calculated
        parts={'amine_head':'CN(C)CCCN','epoxide_tail':epoxide,'acyl_tail':acyl}
        quantities={'amine_head':1,'epoxide_tail':2,'acyl_tail':2}
        first=element_inventory(parts['amine_head'])
        first.update({k:2*v for k,v in element_inventory(epoxide).items()})
        assert first==element_inventory(intermediate)
        full=Counter(first)
        full.update({k:2*v for k,v in element_inventory(acyl).items()})
        right=element_inventory(product);right.update({'H':2,'Cl':2})
        assert full==right
        controls.append({'label':label,'source_locator':'SI PDF p. 3 Fig. S2 full aminoalcohol and DB graphs; p. 23 Table S1 mass entries','complete_components':parts,'quantities':quantities,'expected_stage_1_product':intermediate,'expected_final_product':product,'neutral_formula':rdMolDescriptors.CalcMolFormula(mol),'calculated_neutral_mass':mass,'source_table_calculated':calculated,'source_table_observed':observed,'stage_1_inventory_balanced':True,'full_inventory_with_two_HCl_balanced':True,'analytical_ion_assignment_admitted':False})
    dump(HERE/'control-transcriptions.json',{
        'schema_version':'forge.source_control_transcriptions.v1','seed':0,
        'implementation':pin(ROOT,Path(__file__).resolve()),
        'assets':{'supplement':pin(ROOT,SOURCE/'41467_2024_45537_MOESM1_ESM.pdf'),'article':pin(ROOT,SOURCE/'article.xml')},
        'doi':'10.1038/s41467-024-45537-z','family':'epoxide_opening_o_acylation',
        'source_visual_review':{'supplement_pdf_pages':[3,23],'performed':True},
        'controls':controls,
        'source_procedure':{'locator':'Main Methods: General method for the synthesis of DB-lipidoids','stage_1':{'amine_mmol':0.1,'epoxide_mmol':0.24,'epoxide_equivalents':2.4,'temperature_c':80,'time_h':48},'stage_2':{'DCM_ml':2,'temperature':'room temperature','acyl_chloride_mmol':0.24,'acyl_chloride_equivalents':2.4,'TEA_mmol':0.3,'TEA_equivalents':3,'time_h':12,'workup':'DCM and TEA removed under vacuum; crude products dissolved in EtOH'},'salt_exception':'Excess TEA added in stage 1 when the amine is a salt.'},
        'limitations':['2.4 reagent equivalents per stage do not replace the two incorporated copies.','The source reagent is acyl chloride, not a substituted carboxylic-acid surrogate; net byproduct is two HCl.','Table S1 calculated values match neutral exact masses. Observed values are preserved without imposing an unreported exact ion assignment.','Two epoxide additions constitute one documented stage and two O-acylations the next; no isolated mono-adduct intermediate is claimed.','A source-qualified grouped-stage executor and ambiguity/competing-N tests are still required.'],
        'program_qualification_pass':False,'training_admitted':False,'experimental_execution_admitted':False,
    })


if __name__=='__main__':main()
