#include<vector>
#include<string>
#include<iostream>
#include<iomanip>
#include<sstream>
#include<TFile.h>
#include<TTreeReader.h>
#include<TTreeReaderValue.h>
#include<TTreeReaderArray.h>

void hit_2cell(){
    
    TFile* file = TFile::Open("/home/lhl/tau_decay_pi_pi0_nu.root","read");
    TFile* file1 = new TFile("cell.root","recreate");
    TTree* tree = new TTree("granule","each CsI crystal");
    TTreeReader* treeReader = new TTreeReader("rec",file);
    int num = 6472;
    std::vector<double> east,barrel,west,cell;
    std::vector<double> contain;
    contain.reserve(6500);
    std::string str = "cell_";
    stringstream ss;
    for(int i = 0; i < num; ++i){
        ss << i;
        string str1 = str + ss.str();
        tree->Branch(str1.c_str(),&(contain[i]),"str1.c_str()/D");
        ss.str("");
    }

    TTreeReaderValue<int>* nRecEmcHits = new TTreeReaderValue<int>(*treeReader, "nRecEmcHits");
    TTreeReaderArray<int>* bc = new TTreeReaderArray<int>(*treeReader , "emcHit_bc");
    TTreeReaderArray<int>* id_theta = new TTreeReaderArray<int>(*treeReader, "emcHit_id_theta"); 
    TTreeReaderArray<int>* id_phi = new TTreeReaderArray<int>(*treeReader, "emcHit_id_phi");
    TTreeReaderArray<double>* energy = new TTreeReaderArray<double>(*treeReader, "emcHit_energy");
    TTreeReaderArray<double>* pos_theta = new TTreeReaderArray<double>(*treeReader, "emcHit_pos_theta");
    TTreeReaderArray<double>* pos_phi = new TTreeReaderArray<double>(*treeReader, "emcHit_pos_phi");

    while(treeReader->Next()){      
    east.clear();
    barrel.clear();
    west.clear();
    cell.clear();
    east.assign(576,0.0);
    barrel.assign(5280,0.0);
    west.assign(576,0.0);
    for(int ih = 0; ih < *(*nRecEmcHits); ++ih){
        std::cout << "theta phi = " << (*id_theta)[ih] << "  " <<(*id_phi)[ih] << std::endl;  
      if((*bc)[ih] == 0){
          int count1 = 96 * (*id_theta)[ih] + (*id_phi)[ih]; 
          east[count1] += (*energy)[ih];   
      }
      if((*bc)[ih] == 1){
          int count2 = 120 * (*id_theta)[ih] + (*id_phi)[ih];   
          barrel[count2] += (*energy)[ih];
      }
      if((*bc)[ih] ==2 ){
          int count3 = 96 * (*id_theta)[ih] + (*id_phi)[ih];    
          west[count3] += (*energy)[ih];
      }
    }
    std::cout << "========================================="<< std::endl;
        cell.insert(cell.end(),east.begin(),east.end());
        cell.insert(cell.end(),barrel.begin(),barrel.end());
        cell.insert(cell.end(),west.begin(),west.end());
        for(int i = 0; i < num; i++){
        contain[i] = cell[i];
        }
        tree->Fill(); 
    }
        tree->Write();   
}
