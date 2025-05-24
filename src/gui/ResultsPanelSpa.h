#ifndef gui_ResultsPanelSpa_h_
#define gui_ResultsPanelSpa_h_

class ResultsPanelSpa : public ResultsPanel {
  public:
    ResultsPanelSpa(wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize(500, 300), long style = wxTAB_TRAVERSAL);
    void OnResultsBookPageChanged(wxBookCtrlEvent& event);
};

#endif