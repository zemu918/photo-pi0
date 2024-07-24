
void read_cell(){
    
    TFile* file = TFile::Open("cell_pi_pi0_nu.root","READ");
    TTreeReader* treeReader = new TTreeReader("granule",file);
    TH2F* id_map = new TH2F("id_map", "", 120, 0, 120, 56, 0, 56);

    std::vector<TTreeReaderValue<double>*> TR;
    for (int i = 0; i < 6720; i++){
        TR.push_back(new TTreeReaderValue<double>(*treeReader,Form("cell_%i",i)));
            
    }
    int control = 0;
    while (treeReader->Next()){
        for (int ithe = 0; ithe < 56; ithe++){
            for (int iphi = 0; iphi < 120; iphi++){
                int count = ithe * 120 + iphi;
        
                id_map->Fill( iphi, ithe, *(*(TR[count])));
            }        
        }
    control ++;
    if (control == 1) break; 
    }
    TCanvas* c1 = new TCanvas( "c1", "", 600, 500);
    id_map->Draw("colz");
}
