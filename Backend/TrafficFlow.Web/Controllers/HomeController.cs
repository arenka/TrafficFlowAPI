using Microsoft.AspNetCore.Mvc;

namespace TrafficFlow.Web.Controllers;

public class HomeController : Controller
{
    private readonly IConfiguration _config;

    public HomeController(IConfiguration config)
    {
        _config = config;
    }

    public IActionResult Index()
    {
        ViewBag.ApiBaseUrl = _config["TrafficApiBaseUrl"] ?? "http://localhost:5100";
        return View();
    }
}
