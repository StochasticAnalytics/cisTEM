#ifndef _gui_ResultsPanelTm_h_
#define _gui_ResultsPanelTm_h_

class ResultsPanelTm : public ResultsPanel {
  public:
    ResultsPanelTm(wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize(500, 300), long style = wxTAB_TRAVERSAL);
    void OnResultsBookPageChanged(wxBookCtrlEvent& event);
};

#endif